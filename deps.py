"""Optional third-party imports, guarded so a missing package fails loudly and late."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:                                            # pragma: no cover
    Chem = rdFingerprintGenerator = MurckoScaffold = None
    HAS_RDKIT = False

try:
    from ogb.graphproppred import PygGraphPropPredDataset
    HAS_OGB = True
except ImportError:                                            # pragma: no cover
    PygGraphPropPredDataset = None
    HAS_OGB = False

try:
    from tqdm import tqdm
except ImportError:                                            # pragma: no cover
    def tqdm(it, **kw):
        return it


# Minimum RDKit that this pipeline's stored results were produced with. Installing
# PyTDC WITHOUT --no-deps silently downgrades rdkit (its pin is rdkit<2024.3.1),
# and RDKit decides two things that move results with no error message:
# Chem.CanonicalRankAtoms breaks ties in GraphGrid's node ordering, and
# MurckoScaffoldSmiles / GetScaffoldForMol define every scaffold split. A warning
# here is much cheaper than discovering it in a results table.
RDKIT_MIN = (2024, 3)


def _check_environment():
    """
    Catch the quiet half of the HPC module/conda collision.

    `module load python/...` exports PYTHONHOME and PYTHONPATH, and both apply to
    whatever interpreter runs next -- including conda's. PYTHONHOME kills the
    interpreter outright before any of this code runs ("No module named
    'encodings'"), so it cannot be caught here; it is checked anyway in case only
    a stale value survives. PYTHONPATH is the dangerous one: it prepends another
    Python's site-packages to sys.path without any error, so imports can resolve
    to a different build of numpy, torch or rdkit than the one that was
    installed. With rdkit that silently changes scaffold splits.
    """
    import os
    import sys

    msgs = []
    if os.environ.get("PYTHONHOME"):
        msgs.append(
            f"PYTHONHOME is set to {os.environ['PYTHONHOME']}. Unset it; with conda "
            "it points the interpreter at another Python's standard library.")

    pp = os.environ.get("PYTHONPATH")
    if pp:
        outside = [p for p in pp.split(os.pathsep)
                   if p and not p.startswith(sys.prefix)]
        if outside:
            msgs.append(
                f"PYTHONPATH points outside this environment ({outside[0]}). It can "
                "shadow installed packages with another Python's builds; `unset "
                "PYTHONPATH` unless you set it deliberately. See HPC.md.")

    for mod in ("numpy", "rdkit", "torch"):
        try:
            m = __import__(mod)
            path = getattr(m, "__file__", "") or ""
            if path and not path.startswith(sys.prefix) and "site-packages" in path:
                msgs.append(
                    f"{mod} is being imported from {path}, outside this "
                    f"environment ({sys.prefix}). Something on sys.path is "
                    "shadowing the installed copy.")
        except Exception:
            pass
    return msgs


def check_versions(verbose: bool = True):
    """Warn about dependency versions and environments that can change results."""
    msgs = _check_environment()
    if HAS_RDKIT:
        try:
            import rdkit
            parts = rdkit.__version__.replace(".", " ").split()
            got = (int(parts[0]), int(parts[1]))
            if got < RDKIT_MIN:
                msgs.append(
                    f"rdkit {rdkit.__version__} is older than "
                    f"{RDKIT_MIN[0]}.{RDKIT_MIN[1]}. PyTDC pins rdkit<2024.3.1, so a "
                    "plain `pip install PyTDC` downgrades it. Reinstall with "
                    "`pip install --no-deps PyTDC` and restore rdkit -- scaffold "
                    "splits and GraphGrid node ordering both depend on it.")
        except Exception:
            pass
    try:
        import transformers
        parts = transformers.__version__.split(".")
        if (int(parts[0]), int(parts[1])) < (4, 51):
            msgs.append(
                f"transformers {transformers.__version__} predates 4.51. "
                "MoLFormer's remote code imports transformers.masking_utils, added "
                "around 4.51, so loading the model will fail with an ImportError "
                "for a module you never referenced. `pip install -U "
                "'transformers>=4.51'`, or pin --lm_revision.")
    except ImportError:
        pass
    except Exception:
        pass

    if verbose:
        for m in msgs:
            print(f"  WARNING: {m}")
    return msgs


# The two suites, matching MLCIL/benchmarking_molecular_models' config/dataset/.
OGB_SUITE = ["ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molhiv",
            "ogbg-molsider", "ogbg-moltox21"]

TDC_SUITE = ["AMES", "Bioavailability_Ma", "CYP1A2_Veith", "CYP2C19_Veith",
             "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Veith",
             "CYP2D6_Substrate_CarbonMangels", "CYP2D6_Veith",
             "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Veith", "DILI", "HIA_Hou",
             "PAMPA_NCATS", "Pgp_Broccatelli", "SARSCoV2_3CLPro_Diamond",
             "SARSCoV2_Vitro_Touret", "hERG", "hERG_Karim"]

SUITES = {"ogb": OGB_SUITE, "tdc": TDC_SUITE, "all": OGB_SUITE + TDC_SUITE}

ALL_DATASETS = TDC_SUITE


def normalize_name(n: str) -> str:
    n = n.strip()
    return n if n.startswith("ogbg-") else f"ogbg-{n}"


def resolve_datasets(suite, dataset):
    """
    Turn --suite / --dataset into a validated list of dataset names.

    Accepts a suite name, a list of names, or a single name as a bare string.
    The string case matters: iterating a str yields CHARACTERS, so a bare
    "molbace" that slips through silently becomes eight one-letter datasets and
    fails deep inside the OGB loader with "Invalid dataset name ogbg-m". Names
    are validated here instead, against both registries, so a typo is reported
    with the valid options rather than as a loader error.
    """
    if suite:
        return list(SUITES[suite])
    if not dataset:
        return list(TDC_SUITE)
    if isinstance(dataset, str):
        dataset = [dataset]

    from tdc_data import canonical_tdc_name

    ogb_lower = {n.lower(): n for n in OGB_SUITE}
    out, bad = [], []
    for raw in dataset:
        name = str(raw).strip()
        if canonical_tdc_name(name):
            out.append(name)
        elif name.lower() in ogb_lower:
            out.append(ogb_lower[name.lower()])
        elif f"ogbg-{name.lower()}" in ogb_lower:
            # accept the short form, e.g. "molbace" -> "ogbg-molbace"
            out.append(ogb_lower[f"ogbg-{name.lower()}"])
        else:
            bad.append(name)

    if bad:
        import difflib
        known = OGB_SUITE + [n.replace("ogbg-", "") for n in OGB_SUITE] + TDC_SUITE
        near = []
        for b in bad:
            near += difflib.get_close_matches(b, known, n=3, cutoff=0.6)
        msg = [f"unknown dataset(s): {', '.join(bad)}"]
        if near:
            msg.append(f"did you mean: {', '.join(dict.fromkeys(near))}?")
        elif any(len(b) <= 2 for b in bad):
            msg.append("a one- or two-character name usually means --dataset was "
                       "passed a string that got split; quote it or check the "
                       "argument order.")
        msg.append(f"OGB: {', '.join(OGB_SUITE)}")
        msg.append(f"TDC: {', '.join(TDC_SUITE)}")
        raise SystemExit("\n  ".join(msg))
    return out

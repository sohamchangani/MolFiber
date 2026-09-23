"""
The frozen MolFormer view  m_x = P_F(MolFormer(x)).

IBM's MoLFormer-XL, loaded from the HuggingFace hub and mean-pooled over
non-padding tokens. The model is frozen and every embedding is computed once and
cached to .npy, so the cost is paid once per dataset no matter how many training
runs or search trials follow.

The cache is validated against a signature of the molecule list, not just its row
count: two different molecule sets of equal length would otherwise pass, and the
mismatch would surface much later as molecules paired with a neighbour's
embedding.

A note on trust_remote_code. MoLFormer is not a built-in architecture, so
from_pretrained downloads modelling code from the Hub and imports it. That code
imports transformers internals, so a mismatch between the Hub code and the
INSTALLED transformers shows up as a puzzling ImportError for a module you never
referenced -- most commonly `transformers.masking_utils`, which older
transformers does not have. `from transformers import AutoModel` succeeding tells
you nothing about this, because the failing import happens inside the downloaded
module. Pin --lm_revision to freeze the Hub code, or upgrade transformers.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from deps import tqdm
from views import smiles_signature


def _cache_path(name, args):
    if not args.cache_dir:
        return None
    os.makedirs(args.cache_dir, exist_ok=True)
    tag = args.lm_model.replace("/", "_")
    return os.path.join(args.cache_dir, f"{name}_{tag}.npy")


def molformer_embeddings(smiles, model_name, batch=64, device="cpu", revision=None):
    """Frozen MoLFormer-XL embeddings, mean-pooled over non-padding tokens."""
    from transformers import AutoModel, AutoTokenizer
    kw = {"trust_remote_code": True}
    if revision:
        kw["revision"] = revision
    tok = AutoTokenizer.from_pretrained(model_name, **kw)
    mdl = AutoModel.from_pretrained(model_name, deterministic_eval=True,
                                    **kw).to(device).eval()
    out = []
    with torch.no_grad():
        for s in tqdm(range(0, len(smiles), batch), desc="    molformer", leave=False):
            enc = tok(list(smiles[s:s + batch]), padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    return torch.cat(out).float()


def _load_npy(path, smiles):
    emb = np.load(path)
    if emb.ndim != 2 or emb.shape[0] != len(smiles):
        raise ValueError(f"{path} has shape {emb.shape}, expected ({len(smiles)}, D)")
    sig = path + ".sig"
    if os.path.exists(sig):
        with open(sig) as fh:
            if fh.read().strip() != smiles_signature(smiles):
                raise ValueError(f"{path} was built on a different molecule set")
    return torch.from_numpy(emb.astype(np.float32))


def _diagnose(exc, args):
    """Turn a from_pretrained failure into something actionable."""
    try:
        import transformers
        ver = transformers.__version__
    except Exception:
        return ("transformers is not installed. `pip install \"transformers>=4.51\"`")

    text = f"{type(exc).__name__}: {exc}"
    lines = [f"{text}", f"installed transformers: {ver}"]

    if isinstance(exc, ImportError) and "transformers." in str(exc):
        missing = str(exc).split("'")[-2] if "'" in str(exc) else "a transformers module"
        lines += [
            "",
            "This import comes from MoLFormer's REMOTE CODE, not from this package.",
            "trust_remote_code=True downloads modelling code from the Hub, and that",
            f"code imports {missing}, which this transformers version does not have.",
            "That is why `from transformers import AutoModel` works while loading the",
            "model does not.",
            "",
            "Fix, in order of preference:",
            "  1. pip install -U 'transformers>=4.51'",
            "  2. pin the Hub code to a revision that matches your transformers:",
            "       --lm_revision <commit-sha>",
            "  3. clear stale downloaded modules, which can outlive an upgrade:",
            "       rm -rf ~/.cache/huggingface/modules/transformers_modules",
        ]
    elif "connect" in text.lower() or "offline" in text.lower() or "resolve" in text.lower():
        lines += [
            "",
            "This looks like a network failure. HPC compute nodes often have no",
            "outbound access. Precompute the embeddings on a login node, then run",
            "the job offline against the cache:",
            "",
            "  # login node",
            f"  python main.py --dataset {args.dataset[0] if args.dataset else 'AMES'} "
            "--method rf --cache_dir cache",
            "  # compute node",
            "  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        ]
    return "\n".join(lines)


def build_lm(name, smiles, args, dev):
    """Frozen MolFormer embeddings (N, D), or None if disabled/unavailable."""
    if args.lm == "none":
        return None

    cache = _cache_path(name, args)
    if cache and os.path.exists(cache) and not getattr(args, "rebuild_cache", False):
        try:
            emb = _load_npy(cache, smiles)
            print(f"    molformer: cached {tuple(emb.shape)}")
            return emb
        except ValueError as exc:
            print(f"  WARNING: ignoring stale cache ({exc}); recomputing.\n"
                  "           If this machine cannot load MoLFormer, the recompute "
                  "will fail. Regenerate the\n           cache where it can, with "
                  "--precompute_lm, and copy both the .npy and the .npy.sig.")

    try:
        emb = molformer_embeddings(smiles, args.lm_model, args.lm_batch, dev,
                                   getattr(args, "lm_revision", None))
    except Exception as exc:
        msg = _diagnose(exc, args)
        if getattr(args, "lm_optional", False):
            print(f"  WARNING: {msg}\n  --lm_optional is set, so continuing with "
                  "the GraphGrid view only. MolRouter will be SKIPPED and any "
                  "MolFiber numbers below come from one view, not two.")
            return None
        raise SystemExit(f"\nMolFormer could not be loaded.\n\n{msg}\n")

    if cache:
        np.save(cache, emb.numpy())
        with open(cache + ".sig", "w") as fh:
            fh.write(smiles_signature(smiles))
    print(f"    molformer: {tuple(emb.shape)}")
    return emb

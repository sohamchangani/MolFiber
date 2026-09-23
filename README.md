# MolFiber
From Global Anchors to Local refinement in Molecular Learning.

MolFiber is a framework for combining heterogeneous molecular representations for
property prediction. Rather than fusing views symmetrically, it treats a Morgan
fingerprint as an *anchor* that both makes a prediction and organises chemical space
into local neighbourhoods, and lets the auxiliary views — **GraphGrid** (topological)
and **MolFormer** (learned chemistry) — act only through comparisons made *inside*
those neighbourhoods. The correction they produce is bounded, and the model begins
exactly at the anchor, so the auxiliary views can revise the Morgan prediction but
never overwhelm it.

### Key Features

- **Anchored refinement**: a frozen Morgan random forest supplies the base prediction
  $b_M(x)$; everything else is a bounded residual $\lambda\tanh(\cdot)$ on top of it.
  The readout is zero-initialised, so $S(x) = b_M(x)$ identically at step zero and the
  anchor is a floor rather than a starting point.
- **Approximate Morgan fibers**: each molecule is compared against its $k$ nearest
  Morgan–Tanimoto neighbours, always retrieved from training data, so the method stays
  inductive and no test molecule is compared against another test molecule.
- **Fiber-relative views**: auxiliary information reaches the output only through
  in-fiber pairwise comparisons, never directly — which is what prevents the model
  from collapsing into an ordinary global classifier that ignores the anchor.
- **A shrinkage family, not competing designs**: decomposing the contextual score into
  within-fiber and between-fiber components shows that the absolute (**MolFiber-G**)
  and neighbourhood-centred (**MolFiber-L**) corrections are the two endpoints of one
  family. **MolFiber-M** fits the shrinkage between them as a per-target coefficient;
  **MolFiber-A** sets it per molecule in closed form by empirical Bayes. Both nest the
  endpoints exactly.
- **Benchmark-faithful evaluation**: OGB and TDC datasets under each benchmark's own
  prescribed split, with the ADMET benchmark-group protocol reproduced exactly,
  including its merged training pool and inner cross-validation.
- **Fusion baselines included**: early, late and gated fusion of the same three views,
  plus an anchored-MLP ablation that removes the fiber and holds everything else fixed.

### Methodology

**The pipeline**

1. **Three views.** Morgan fingerprint $M(x)$ (ECFP4), GraphGrid image $G(x)$, and
   MolFormer embedding $F(x)$.
2. **Anchor.** A random forest on $M(x)$ gives $b_M(x)$, cross-fitted out of fold on
   training molecules so no molecule's own label enters its anchor score.
3. **GraphGrid.** Vertices are scored by four purely topological descriptors (heat
   kernel signature, degree centrality, $k$-core number, PageRank), quantile-binned,
   sorted lexicographically, and the reordered adjacency is block-pooled into a fixed
   $\kappa \times \kappa$ image. No atom types or bond orders enter, so this view
   carries shape only.
4. **Fibers.** $\mathcal{B}_x = \{x\} \cup \mathrm{kNN}_M(x; \mathcal{D}_\mathrm{train})$.
5. **Pair encoding.** Every ordered pair in the fiber is encoded from
   $[g_z - g_w,\ g_z \odot g_w,\ m_z - m_w,\ m_z \odot m_w,\ s_M(z,w)]$, aggregated
   with similarity-weighted attention, and read out to a contextual score.
6. **Correction.** $S(x) = b_M(x) + \lambda\tanh(u_{x,x} - (1-\rho)\,\bar{u}_x)$, with
   $\rho$ fixed at $1$ (G) or $0$ (L), fitted per target (M), or estimated per molecule
   (A).

### Results and Efficiency

MolFiber is evaluated on 24 benchmarks spanning OGB and TDC under each benchmark's
prescribed scaffold split. For results and comparison to baselines, refer to Table 1
in our paper.

Feature construction (fingerprints, GraphGrid images, MolFormer embeddings, fiber
similarity blocks) is a one-off cost per dataset, cached and shared across variants,
seeds and baselines. The dominant per-epoch cost is the pair encoder, which evaluates
all $(k+1)^2$ ordered pairs per fiber — so runtime grows quadratically in fiber size,
not linearly. `benchmark_runtime.py` reproduces the timing tables.

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps PyTDC
```

PyTDC is installed with `--no-deps` deliberately. Its pins downgrade RDKit below
`2024.3.1`, and RDKit feeds both GraphGrid's node-order tie-break and every scaffold
split — so a silent downgrade can move results with no error message at all. See the
comments in `requirements.txt` for the full reasoning. Python 3.11 or 3.12 is the
smoother path.

### Usage

```bash
# the two benchmark suites
python main.py --suite tdc --method combine --repeat 5
python main.py --suite ogb --method combine --repeat 5

# a single dataset, one variant
python main.py --dataset AMES --method molfiber_l
python main.py --dataset ogbg-molbace --method molfiber_a

# fusion ablations (standalone; imports nothing from the main pipeline)
python run_ablations.py --suite tdc --repeat 5

# runtime benchmark (fully self-contained single file)
python benchmark_runtime.py --dataset ogbg-molbace --epochs 5
```

### Repository layout

| File | Contents |
|------|----------|
| `main.py` | Entry point |
| `cli.py` | All command-line options |
| `runner.py` | Per-dataset driver: load, split, anchor, views, train, report |
| `model.py` | `MolFiber` (G/L/M/A), `MolRouter`, `GraphGridCNN` |
| `graphgrid.py` | Topological descriptors, canonical vertex ordering, block pooling |
| `lm.py` | Frozen MolFormer embeddings, cached |
| `fibers.py` | k-NN retrieval and within-fiber similarity blocks |
| `fingerprints.py` | Multi-fingerprint featurisation and similarities |
| `anchor.py` | Cross-fitted Morgan random forest |
| `splits.py` | OGB, TDC and the KANO / TOPOFORMER scaffold split ports |
| `tdc_data.py` | TDC loading, ADMET benchmark-group splits |
| `innercv.py` | Inner cross-validation for protocols with no validation fold |
| `train.py` | Training and prediction loops |
| `ablation_models.py`, `run_ablations.py` | Fusion baselines, standalone |
| `benchmark_runtime.py` | Self-contained runtime benchmark |

### Requirements

Python 3.11 or 3.12 recommended (3.13 works, but more packages build from source).

| Package | Version | Description |
|---------|---------|-------------|
| python | >= 3.9 | Core language (3.8 cannot use the MolFormer view) |
| torch | >= 2.0, < 3.0 | Deep learning framework |
| numpy | >= 1.26, < 2.3 | Numerical computing |
| pandas | >= 2.0, < 3.0 | Dataset tables and split mapping |
| scikit-learn | >= 1.3, < 1.8 | Random-forest anchor and ROC-AUC |
| rdkit | >= 2024.3 | Fingerprints, scaffolds, canonical atom ranking |
| transformers | >= 5.0 | MolFormer checkpoint (current Hub revision needs the 5.x API) |
| ogb | >= 1.3.6 | OGB datasets and their scaffold splits |
| torch_geometric | >= 2.4, < 3.0 | Graph data structures for OGB |
| PyTDC | latest | TDC datasets (install with `--no-deps`) |
| tqdm | >= 4.65 | Progress bars |


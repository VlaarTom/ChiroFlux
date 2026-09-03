# ChiroFlux

Collective-variable creation and analysis for TIS/RETIS path sampling with
[infretis](https://github.com/infretis/infretis).

Given a simulation — its `.toml` config, its `infretis_data.txt` path table and a
folder of per-path CV trajectories — ChiroFlux answers *which collective
variables actually decide the outcome*: which CVs separate reactive from
non-reactive paths, or one simulation from another (e.g. two enantiomers).

Every analysis is WHAM-weighted, so paths sampled in different TIS ensembles are
combined without bias, and everything is resolved **per interface**

## Install

```bash
pip install -e .
```

`chiroflux train-deeptda` additionally needs torch, lightning and mlcolvar,
which are kept out of the base install because they are large:

```bash
pip install -e '.[deeptda]'      # add the DeepTDA stack
pip install -e '.[dev]'          # pytest + ruff
```

`generate-cvs` is the only command that reads MD trajectories; everything else
works from the `.txt` files it produces. It needs the trajectories, the
topology and the `.ndx` index files alongside them.
[mlcolvar](https://mlcolvar.readthedocs.io/en/stable/installation.html)

## Input layout

Most commands expect the standard infretis output plus a folder of per-path CV
trajectories (`-cv-dir`, default `ML/`), one `<path_nr>.txt` per path:

```
line 1: # reactive          |  # non-reactive
line 2: # <ensemble info>
line 3: # <duplicate/edit info>
line 4: <column names>            (no leading '#')
line 5+: <data>
```

One column is the order parameter (`-op-col`, default `OP_Lamb`); the rest are
treated as CVs unless narrowed with `-cv-cols` / `-exclude`.

## Commands

```bash
chiroflux --help
chiroflux COMMAND --help
```

| Command | What it does |
| --- | --- |
| `generate-cvs` | Computes the per-frame CVs from the MD trajectories and writes the per-path `.txt` files every other command reads. |
| `histograms` | Weighted CV histograms, statistics and 2D maps over a path ensemble, optionally merging a second simulation onto a common OP axis. Requires a `-ranges` file (see below). |
| `sasa` | Weighted solvent-accessible surface area profile across the membrane, from a Shrake–Rupley construction on the trajectories. Requires a `-runs` file (see below). |
| `membrane-spatial` | Spatial membrane structure around the permeant: radial/z maps, curvature, local thickness and bonded metrics. |
| `neighbours` | Lipid neighbour composition around the permeant per membrane slab, with bootstrap enrichment statistics against bulk composition. |
| `shap-ml` | Fits WHAM-weighted classifiers (random forest, logistic regression, gradient boosting, LightGBM, SVM) per interface and explains them with SHAP. |
| `shap-enantiomer` | Same, but the label is *which of two simulations* a path came from. |
| `statistics` | Model-free weighted effect sizes (Cohen's d, Spearman ρ, KS distance) per interface — a cheap sanity check on the SHAP rankings. |
| `pca` | Weighted PCA of the CV space, optionally on a joint basis fitted across two simulations so they can be compared in the same coordinates. |
| `prepare-deeptda-data` | Builds a frame-level, weighted DeepTDA training set labelled reactive/non-reactive. |
| `prepare-deeptda-data-ld` | Same, but labelled by which of two simulations each frame came from. |
| `train-deeptda` | Trains a 2-state DeepTDA CV on either dataset and reports the input correlations that show which CVs drive it. |

A typical single-simulation run:

```bash
chiroflux shap-ml    -toml infretis.toml -data infretis_data.txt -cv-dir ML
```

Comparing two simulations (e.g. L and D enantiomers). These commands take a
*root directory* per simulation and expect the `.toml`, the data file and the
CV folder inside it:

```bash
chiroflux shap-enantiomer          -dir-l /path/to/L/folder/ -dir-d /path/to/D/folder/
chiroflux prepare-deeptda-data-ld  -dir-l /path/to/L/folder/ -dir-d /path/to/D/folder/
chiroflux train-deeptda -npz deeptda_ld_dataset.npz -class-names L,D
```

`pca` instead takes the second simulation as a parallel set of options
(`-toml2`, `-data2`, `-cv-dir2`) and fits one joint basis over both:

```bash
chiroflux pca -toml L/infretis.toml -cv-dir L/ML \
              -toml2 D/infretis.toml -cv-dir2 D/ML -label1 L -label2 D
```

### GPU SHAP: the `-shap-device` flag

Explaining, not fitting, dominates `shap-ml` and `shap-enantiomer`. On a
300-tree forest, fitting took ~1 s while `TreeExplainer` took ~177 s for a
test fold — **99.4% of the time**. shap ships a CUDA implementation of the same
Tree SHAP algorithm, and `-shap-device` selects it:

| value | behaviour |
| --- | --- |
| `auto` (default) | use the GPU when one is usable, fall back to the CPU explainer with a warning otherwise |
| `gpu` | fail rather than fall back — use when you want to know the GPU is really being used |
| `cpu` | force the multi-core CPU path |

Measured on an RTX 2000 Ada, 300 trees, 600 test rows × 40 CVs:

```
device=cpu (all 22 cores) : 13.58 s
device=gpu                :  3.56 s     x3.8
attributions agree        : True, max|diff| 2.2e-06
```

The difference is float32 on the GPU against float64 on the CPU, so rankings
are unaffected — there is a test asserting the two agree.

The GPU *replaces* the process fan-out rather than adding to it: `-n-jobs`
workers all competing for one card would only contend for it and its VRAM.
Batches are capped at 2000 rows to bound VRAM rather than host RAM.

This applies to `rf` and `gbm`. LightGBM is handed a `DataFrame`, and shap's
GPU TreeSHAP warns that categorical features are unsupported there, so that
model stays on the CPU — it explains in milliseconds either way. `logreg` uses
`LinearExplainer` and `svm` uses permutation importance, so neither is affected.

Note this accelerates *explaining*. Fitting is the other 0.6% — for the SVM,
which is not a tree model and is dominated by permutation importance instead,
see `-svm-device` below.

### GPU SVM: the `-svm-device` flag

`-models svm` spends ~75% of its time in `permutation_importance`, which is
`predict`-bound rather than fit-bound. cuML's SVC accelerates exactly that:

```
                fit        permutation_importance      total
sklearn CPU    1.59 s            192.15 s            193.74 s
cuML GPU       2.73 s              2.44 s              5.17 s     x37.5 overall
ranking agreement (Spearman): 0.9998, identical top-3 CVs
```

`-svm-device` takes `auto` (default), `gpu` or `cpu`, mirroring `-shap-device`.
It needs cuML installed; without it, `auto` falls back to scikit-learn with a
warning and the package behaves exactly as before.

#### Installing the GPU paths

**A new environment** needs no special sequence. `shap`'s GPU TreeSHAP lives in
the CUDA build of the conda package (the PyPI wheels carry no CUDA extension),
so take `shap` from conda-forge and everything else from pip:

```bash
conda create -n chiroflux_gpu -c conda-forge python=3.13 "shap=0.52.0=cuda129*"
conda activate chiroflux_gpu
pip install -e '.[gpu,deeptda]'
```

pip resolves cuML and torch together in one transaction and picks a mutually
compatible CUDA runtime. Verify with:

```bash
python -c "import importlib.util as u; print(u.find_spec('shap._cext_gpu') is not None)"   # -shap-device gpu
python -c "import cuml, torch; print(cuml.__version__, torch.cuda.is_available())"          # -svm-device gpu
pytest
```

chiroflux never imports infretis - it only reads its output files - so this
environment does not need infretis unless you also run simulations from it.

**Migrating an existing environment** is harder, and only because pip will not
renegotiate an already-installed torch. If yours is a cu128 build, pip keeps it
and then takes the newest cuML, which wants CUDA 12.9 - a mismatch that shows
up at runtime as `CUDA_ERROR_INVALID_IMAGE`. In that case:

```bash
conda create --name myenv_gpu --clone myenv          # keep a fallback

pip install --upgrade "torch==2.13.0+cu129" torchvision \
    --index-url https://download.pytorch.org/whl/cu129
pip install -e '.[gpu]'

# orphaned CUDA 13 packages, if any, also cause CUDA_ERROR_INVALID_IMAGE
pip uninstall -y nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc \
    nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile \
    nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx

# those packages share one nvidia/ tree, so the uninstall above deletes files
# torch owns (libcudnn.so.9 among them) - put them back
pip install --force-reinstall --no-deps \
    nvidia-cudnn-cu12==9.20.0.48 nvidia-cusparselt-cu12==0.8.1 \
    nvidia-nccl-cu12==2.29.7 nvidia-nvshmem-cu12==3.4.5
```

Skipping the last step leaves torch broken, so `train-deeptda` stops working
while cuML starts. Run `pytest` and a `torch.cuda.is_available()` check before
trusting a migrated environment - and prefer the fresh environment above, which
needs none of this.

### Histogram binning: the `-ranges` file

`chiroflux histograms` requires `-ranges`, a TOML file giving the binning for
every CV:

```toml
[ranges]
"OP_Lamb" = [-25, -13.5, 125]     # min, max, n_bins
"Mem_APL" = [0.6, 0.9, 60]

[ranges_nr_minus]                 # overrides for ("non-reactive", "minus")
"OP_Lamb" = [-36, -24, 120]
```

Start from [`examples/column_ranges.toml`](examples/column_ranges.toml) (120
columns, the values the original script carried) and copy it per simulation:

```bash
cp examples/column_ranges.toml L/ranges.toml    # then edit the OP window
chiroflux histograms -cv-dir L/ML -weights L/path_weights.txt -ranges L/ranges.toml
```

These ranges are genuinely per-simulation — each run covers a different OP
window — which is why they are an input rather than a constant in the code.
Keeping them inline is what produced four divergent copies of the original
script, differing only in configuration. There is deliberately **no default**:
the binning determines every histogram in the output, so a plausible-but-wrong
fallback would be worse than refusing to run. Malformed entries (wrong length,
`max <= min`, zero bins) are rejected with the offending column named.

### Comparing two SASA profiles

`sasa` combines the runs in its `-runs` file into one profile; `sasa-compare`
takes two such profiles and reports where they differ:

```bash
chiroflux sasa -runs L_runs.toml -out-dir sasa_L
chiroflux sasa -runs D_runs.toml -out-dir sasa_D
chiroflux sasa-compare -a sasa_L -b sasa_D -label-a L -label-b D
```

It recomputes both profiles from the cached per-path arrays, so no trajectory
is re-read and the comparison can be re-run freely while adjusting it.

If the two permeants traverse the membrane in **opposite directions**, their
depth axes run opposite ways and the profiles are not comparable bin for bin.
`-mirror-b` reflects B through the membrane centre (z → −z) first:

```bash
chiroflux sasa-compare -a sasa_L -b sasa_D -label-a L -label-b D -mirror-b
```

On a test profile peaking at z = −20 against one peaking at z = +20, this took
the bins flagged as different from 24/40 down to 3/40 — the rest were an
artefact of the misaligned axis. The reflection also swaps the phosphate-plane
landmarks, since the upper leaflet becomes the lower one.

Two guards: it requires a z range symmetric about zero (reversing bins is only
z → −z there), and it warns if B was already built with a run-level `mirror_z`,
because mirroring twice returns the original.

The difference Δ(z) = B − A is bootstrapped **jointly**: each replicate
resamples paths within A and within B and differences the two resampled means.
Drawing two separate confidence bands and checking whether they overlap is the
intuitive alternative and it is wrong — non-overlapping intervals do imply a
difference, but overlapping ones do not imply its absence, so that reading
misses real effects. As with `sasa`, the resampling unit is the **path**, since
frames within a path are consecutive points of one trajectory.

Output is one plot per quantity (total / polar / apolar SASA and exposed
fraction) showing both profiles above and Δ with its band below, plus
`sasa_comparison.csv` giving per-bin Δ, interval and a significance flag.

Both runs must share a z axis. `sasa` records its binning in `sasa_meta.json`
and the comparison refuses mismatched `-z-range`, `-z-bin-width`,
`-probe-radius`, `-fold-symmetric` or `-occlude-with-water`.

### Which simulations to combine: the `-runs` file

`chiroflux sasa` requires `-runs`, a TOML file with one `[[run]]` table per
simulation. Start from [`examples/sasa_runs.toml`](examples/sasa_runs.toml):

```toml
[[run]]
name     = "entry"
load_dir = "L_PRO_neutral/infinit_entry/load"
weights  = "L_PRO_neutral/infinit_entry/wham/path_weights.txt"
ml_dir   = "L_PRO_neutral/infinit_entry/post/ML"
tpr      = "L_PRO_neutral/infinit_entry/topol.tpr"
scale    = 1.0        # multiplies this run's weighted histogram
mirror_z = false      # reflect z, for a run entered from the other leaflet
```

Runs are summed onto one z axis, so `scale` puts them on a common footing: use
`1.0` for the reference run, and for a second run referenced to its own state A
use the ratio of total crossing probabilities (the last row of each run's
`wham/Pcross.txt`, column 2). The ratio of crossing probabilities is sufficient 
since the paths are weighted in the analysis. `mirror_z` is the SASA-profile 
counterpart of `-flip-*` — set it for a run whose permeant entered from the 
opposite leaflet.

Every path in the file is checked for existence before a single trajectory is
read, and a non-positive `scale` or a duplicate run name is rejected outright.

### Worked example: two simulations entered from opposite leaflets

L and D were run with the permeant entering from opposite sides of the
membrane, so this run has to undo *both* a chirality convention and a
direction-of-entry convention before the classifier sees the data:

```bash
chiroflux shap-enantiomer \
  `# ── which simulations ──────────────────────────────────────────────` \
  -dir-l data/L/ -dir-d data/D/ \
  -data-l infretis_data_19.txt -data-d infretis_data_17.txt \
  \
  `# ── drop CVs that cannot inform an L/D comparison ──────────────────` \
  `# z_O2/z_O3/z_C2/z_C3/z_N/z_P: lipid-atom families kept as controls in CV data ` \
  `# intentially kept manually instead of one flag to exclude those               ` \
  `# ACSF, z_PRO, cen_vec: not discriminating here                                ` \
  -exclude z_O,z_C,z_N,z_P,cen_vec,ACSF,lambda \
  \
  `# ── collapse the leaflet pairs onto one naming scheme ──────────────` \
  `# Keep only the LOWER-leaflet columns of L and the UPPER-leaflet    ` \
  `# columns of D, then rename what survives to a single convention.   ` \
  `# Dropping one side of each pair first is what makes the rename     ` \
  `# safe: -name-cv-cols applies substitutions in order, so it cannot  ` \
  `# express a genuine two-way swap.                                   ` \
  -exclude-l _u_,2u,N_Pu \
  -exclude-d _l_,2l,N_Pl \
  -name-cv-cols _l_:_u_,2l:2u,N_Pl:N_Pu \
  \
  `# ── symmetry corrections, each applied to ONE simulation ───────────` \
  `# theta -> 180 - theta: unsigned angles vs the membrane normal,     ` \
  `# whose +z/-z face is swapped by the opposite entry direction.      ` \
  -flip-d PRO_ang_C_CG,PRO_r_plane_chiral \
  `# theta -> -theta: chirality-odd pseudoscalars, negated between     ` \
  `# enantiomers by definition (dihedrals, signed volume, handed CNs). ` \
  -mirror-d PRO_dih_,PRO_sign_vol,PRO_hCN_,PRO_nCos_,PRO_azim_ \
  `# phi -> phi + 180: the Cremer-Pople phase, whose mirror is half a  ` \
  `# pseudorotation cycle away rather than negated.                    ` \
  -phase-shift-l PRO_CP_phi2 \
  \
  `# ── representation ─────────────────────────────────────────────────` \
  `# z_Memb is the reference for the z-corrections; it carries no      ` \
  `# independent signal, so drop it from the feature set afterwards.   ` \
  -drop-z-ref \
  \
  `# ── model, parallelism, output ─────────────────────────────────────` \
  -optimize -n-jobs 28 -O
```

Three things in this example are worth copying deliberately:

- **Corrections are per simulation.** Each one is applied to L *or* D, never
  both — applying the same correction to both cancels it. Which side you
  correct is free (here `-phase-shift-l` brings L into D's frame while the
  other two bring D into L's), as long as each CV is corrected exactly once.
- **Substrings, not names.** `-exclude`, `-mirror-*`, `-flip-*` and
  `-phase-shift-*` match by substring, so `PRO_dih_` catches both
  `PRO_dih_chiral` and `PRO_dih_OH`. Keep them specific: a bare `PRO` would
  match essentially every CV in this feature set. (`-angle-cols`,
  `-sym-angle-cols` and `-z-cols` match **exact** names instead.)
- **`-exclude` runs before the corrections.** A CV dropped there is not
  available to be corrected, so keep the patterns narrow enough not to catch
  something you meant to correct. The `z_`-prefixed patterns above are chosen
  for exactly that reason: they drop the per-atom z columns without touching
  the `PRO_hCN_C2*`/`PRO_nCos_C2*` handed coordination numbers, which a bare
  `C2,O2` would also have removed — and those pseudoscalars are the CVs most
  able to resolve L from D.

  Excluding the z columns also means every entry in the built-in z-correction
  list is now absent, which is reported as one `Skipping z-correction for
  'z_...': dropped by -exclude` line each rather than as a warning.

When copying the block, keep the line continuations clean: a `\` continues a
line only when it is the **last** character. A trailing space turns it into an
escaped space, which arrives as a lone argument and fails with
`Got unexpected extra argument ( )`.

Note that options use a single dash (`-toml`, `-cv-dir`), matching the infretis
tooling convention rather than the GNU `--long-option` style.

`membrane-spatial` and `neighbours` came from scripts that already used
double-dash options, so they accept **both** spellings — `-start` and `--start`
are the same flag — and existing command lines keep working. Their few
multi-value options changed from space- to comma-separated, since that has no
single-dash equivalent:

```bash
chiroflux neighbours -start 1 -slab-range -40,40        # was: --slab-range -40 40
chiroflux membrane-spatial -start 1 -near-n 5,10        # was: --near-n 5 10
```

### Aligning angle conventions between two simulations

A CV that differs between L and D purely by convention will "perfectly"
separate them while carrying no physical information, and will dominate any
importance ranking. `shap-enantiomer` has three corrections for this, grouped
in `--help` under *CV corrections: symmetry*. They fix *different* things, and
which you need depends on how the CV is defined:

| flag | operation | corrects for | applies to |
| --- | --- | --- | --- |
| `-mirror-l` / `-mirror-d` | θ → −θ | **chirality**: a chirality-odd CV is negated between mirror-image enantiomers by definition, so L has φ where D has −φ | **signed** CVs on a zero-centred domain — an `arctan2` dihedral in [−180, 180], a signed volume, a handed coordination number |
| `-flip-l` / `-flip-d` | θ → 180 − θ | **direction of entry/internal/escape**: a permeant entering from the other leaflet sees the membrane normal reversed, swapping the +z (extracellular) and −z (intracellular) face | **unsigned** angles against the membrane normal, e.g. anything from `arccos` in [0, 180] |
| `-phase-shift-l` / `-phase-shift-d` | φ → φ + 180 | **chirality of a periodic phase** whose reference vector is a pseudovector, so the mirror image lies half a cycle away rather than negated | **periodic phases** in (−180, 180] — e.g. a Cremer–Pople puckering phase, whose C3-endo mirrors to C3-exo |

All three are their own inverse, and each should be applied to **one simulation
only**, so its values become comparable to the other's.

`-flip-*` is not a chirality operation — it will not turn L into D. 90°
is its pivot because that is the flat-vs-vertical boundary: below 90° the
reference face points toward +z, above it toward −z.

`-phase-shift-*` exists because neither of the others is right for a puckering
phase: the mirror is φ+180, not −φ and not 180−φ. It wraps back into
(−180, 180] so the shifted simulation stays on the same domain as the
unshifted one, and warns rather than silently re-basing a [0, 360) column.

Match the flag to the domain. Negating an `arccos` angle sends [0, 180] to
[−180, 0], off its own domain; flipping a signed dihedral gives
180 − (−170) = 350, likewise off-domain.

Columns already folded to `cos(...)`/`cos2(...)` by `-angle-cols` are skipped
by `-flip-*` with a warning, since they hold values in [−1, 1] rather
than degrees. (Negating them is the equivalent operation there, because
−cos θ = cos(180 − θ).)

> **Known limitation.** `-name-cv-cols` applies its substitutions in order, so
> it cannot express a two-way swap: `'_u_:_l_,_l_:_u_'` collapses both onto
> `_u_` and silently produces duplicate column names. Leaflet-paired columns
> (`*_u`/`*_l`, `z_*Top`/`z_*Bot`) therefore cannot currently be exchanged when
> the two simulations were entered from opposite sides.

## Library use

Each analysis lives in its own module and is imported from there:

```python
from chiroflux.shap_analysis import shap_ml
from chiroflux.shap_analysis_ld import shap_enantiomer
from chiroflux.statistical_analysis import statistics
from chiroflux.principal_component_analysis import PCA
from chiroflux.prepare_deeptda_data import prepare_deeptda_data, prepare_deeptda_data_ld
from chiroflux.train_deeptda import train_deeptda
```

They are not re-exported from the top-level `chiroflux` namespace on purpose:
two of them share a name with their own module, so an alias would resolve to the
function or the module depending on import order. Importing from the module is
unambiguous, and it keeps `import chiroflux` from pulling in shap, scikit-learn,
lightgbm and torch.

The CLI loads lazily for the same reason — `chiroflux --help` imports nothing
beyond typer, and a subcommand imports only the module it needs.

### Support modules

The seven entry points sit on three support modules, which import nothing from
the rest of the package:

| module | holds | depends on |
| --- | --- | --- |
| `pathdata.py` | Reading a simulation: the `.toml`, the `infretis_data.txt` path table, the per-path trajectories, column discovery, and the WHAM path weights derived from them. Plus `_check_overwrite`. | numpy, tomli |
| `cvs.py` | Transforms on an already-loaded CV matrix: angle→cos folding, z re-referencing, enantiomer mirrors / entry flips / renames, frame subsampling. | numpy |
| `plotting.py` | The two plots shared between analyses: `_plot_importance_bar` and `_plot_interface_heatmap`. | numpy, matplotlib |

`_compute_path_weights` is arithmetic rather than I/O, but lives in `pathdata`
because it is never useful alone — every caller reads the path table and
immediately weights it.

**None of these import shap, scikit-learn or lightgbm, and that is load-bearing.**
It is what lets `statistics`, `pca`, `prepare-deeptda-data` and `train-deeptda`
run without the ML stack: PCA used to spend 2.0 s loading gradient boosting to
get `_load_path_table`, and now imports in 0.46 s. Only `shap_analysis` and
`shap_analysis_ld` pull that stack in, which is what they are for. Tests in
`tests/test_packaging.py` enforce this.

The SHAP-specific plots (beeswarm, dependence, ROC, calibration) stay in
`shap_analysis.py` rather than moving to `plotting.py` for the same reason —
they need shap and scikit-learn, and hoisting them would hand that cost to
every importer of `plotting`.

## Development

```bash
pytest          # packaging/CLI wiring, shared helpers, weighting maths
ruff check .
```

## Known issue

`_weighted_spearman` in `statistical_analysis.py` does not average tied ranks:
`np.argsort(np.argsort(x))` gives a constant column the distinct ranks
`0..n-1`, so its `std_r < 1e-12` guard can never fire and a degenerate CV
reports a large spurious ρ instead of 0. Covered by an `xfail` test in
`tests/test_weighting.py`. Cohen's d and the KS distance are unaffected.

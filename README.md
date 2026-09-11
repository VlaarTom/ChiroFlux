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

`chiroflux train-deeptda` additionally needs torch, lightning and [mlcolvar](https://mlcolvar.readthedocs.io/en/stable/),
which are kept out of the base install because they are large:

```bash
pip install -e '.[deeptda]'      # add the DeepTDA stack
pip install -e '.[dev]'          # pytest + ruff
```

`generate-cvs` is the only command that reads MD trajectories; everything else
works from the `.txt` files it produces. It needs the trajectories, the
topology and the `.ndx` index files alongside them.

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
| `permeant-index` | Writes the permeant-side `.ndx` files, checking each group's atoms are bonded the way its CV assumes. |
| `leaflet-index` | Writes the `CN_*.ndx` leaflet groups, assigning each lipid to a leaflet by its own headgroup rather than by each atom's height. |
| `histograms` | Weighted CV histograms, statistics and 2D maps over a path ensemble, optionally merging a second simulation onto a common OP axis. Requires a `-ranges` file (see below). |
| `sasa` | Weighted solvent-accessible surface area profile across the membrane, from a Shrake–Rupley construction on the trajectories. Requires a `-runs` file (see below). |
| `membrane-spatial` | Spatial membrane structure around the permeant: radial/z maps, curvature, local thickness and bonded metrics. |
| `preference-compare` | Difference the DOPC/POPC contact preference of two simulations along the OP, with a bootstrapped CI on the difference. |
| `neighbours` | Lipid neighbour composition around the permeant per membrane slab, with bootstrap enrichment statistics against bulk composition. |
| `shap-ml` | Fits WHAM-weighted classifiers (random forest, logistic regression, gradient boosting, LightGBM, SVM) per interface and explains them with SHAP. |
| `shap-enantiomer` | Same, but the label is *which of two simulations* a path came from. |
| `statistics` | Model-free weighted effect sizes (Cohen's d, Spearman ρ, KS distance) per interface — a cheap sanity check on the SHAP rankings. |
| `dynamics` | Time-resolved CV features in a window around each crossing: fluctuation amplitude, autocorrelation time, and lead/lag event ordering against the order parameter. Optional FFT features, with a built-in redundancy check against them. |
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
  `# Every leaflet-marked column spells the leaflet as a MEDIAL field, so   ` \
  `# one pattern reaches all of them: CA_C2_u_DOPC, PRO_hCN_u_P,           ` \
  `# PRO_hCN_u_C2_DOPC_s1, Mem_u_tilt, Mem_u_def. Do not shorten to a bare ` \
  `# "_l": that also matches Mem_thick_loc, which is not leaflet-paired,   ` \
  `# and would rename it to Mem_thick_uoc.                                  ` \
  -exclude-l _u_ \
  -exclude-d _l_ \
  -name-cv-cols _l_:_u_ \
  \
  `# ── symmetry corrections, each applied to ONE simulation ───────────` \
  `# theta -> 180 - theta: unsigned angles vs the membrane normal,     ` \
  `# whose +z/-z face is swapped by the opposite entry direction.      ` \
  -flip-d PRO_ang_C_CG,PRO_r_plane_chiral,PRO_mode_,_tilt \
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

#### Which correction the geometry-resolved chirality CVs need

`generate-cvs` emits three kinds of quantity around the permeant stereocentre,
and they do **not** all take the same correction:

| CVs | kind | correction |
| --- | --- | --- |
| `PRO_hCN_*`, `PRO_hCN_*_s1..s3`, `PRO_nCos_*`, `PRO_azim_*`, `PRO_sign_vol`, `PRO_dih_chiral` | signed **pseudoscalar**, zero-centred | `-mirror-l` / `-mirror-d` |
| `Mem_u_tilt`, `Mem_l_tilt`, `PRO_mode_ring`, `PRO_mode_O`, `PRO_mode_N` | cosine of a **true vector** against +z | `-flip-l` / `-flip-d` (negation, on a cos-valued column) |
| `PRO_rMin_*`, `Mem_u_tiltN`, `Mem_l_tiltN` | true scalar (a distance, a count) | **none** — these are reflection-invariant |

Mirroring the handed columns is right and does not throw the chirality signal
away. To leading order the lipid environment is achiral, so an L path's `hCN`
distribution is the mirror of a D path's and the bulk of the L/D difference in
these columns reflects that near-symmetry rather than any discrimination.
`-mirror-*` removes exactly that leading antisymmetry; what survives is the
diastereomeric residual, which is the quantity of interest.

What must **not** be done to them is `-sym-angle-cols`. Folding to cos²θ
discards the sign irreversibly, taking the residual with it — unlike negation,
which is invertible. Reserve it for genuinely head–tail symmetric angles.

> **Known limitation.** `-name-cv-cols` applies its substitutions in order, so
> it cannot express a two-way swap: `'_u_:_l_,_l_:_u_'` collapses both onto
> `_u_` and silently produces duplicate column names. Therefore, the individual
> simulations should already exclude the opposite leaflet with no interaction.
> **Have to come up with a fix for the internal simulations.**

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

## Known issues

### Leaflet labels are unreliable for the deep chain carbons

The `CN_*.ndx` leaflet groups are cut by a static z threshold evaluated at
frame 0. For the headgroup, glycerol and ester markers that is exact — C2, P,
N, O22 and O32 all split 55/55 in DOPC and 11/11 in POPC. For the deep chain
carbons it is not: DOPC C210 splits 54/56 and C310 51/59, and POPC C210 10/12.

This is **not** lipid flip-flop. A translocated lipid would carry its whole
headgroup across, and the P/N/C2 counts would be off by the same number; they
are exact. What the cutoff catches is chain **interdigitation** — C10 sits deep
in the hydrophobic core, and chain ends from opposing leaflets reach across the
midplane. The species dependence confirms it: DOPC's sn-3 chain is oleoyl
(18:1, long and kinked) and misassigns four carbons, while POPC's is palmitoyl
(16:0) and misassigns none. A flip-flop rate would not care which carbon you
looked at.

Consequence: the 24 coordination numbers built on groups 11–14
(`CA_CC2_*`, `CA_CC3_*`, `N_CC2_*`, `N_CC3_*`, `O_CC2_*`, `O_CC3_*`) count a
few percent of the opposite leaflet's chain carbons — and, since the
misassigned atoms are exactly those nearest the midplane, that contamination is
concentrated where a permeant crossing the core actually is. `Mem_*_tilt`
avoids the problem by reassigning chain atoms geometrically rather than by
label (see `compute_local_chain_tilt`); the coordination numbers do not.

The error is static and shared across paths *within* a simulation, so it should
largely cancel in reactive-vs-non-reactive comparisons. **It does not
automatically cancel between two simulations**: L and D are built by the same
script, but from their own frame-0 structures, so the misassigned set differs
between them and the affected CVs carry a small systematic L/D difference that
has nothing to do with chirality.

**The fix is `chiroflux leaflet-index`**, which assigns each lipid to a leaflet
once from its own headgroup phosphorus and then writes every marker of that
residue to the matching group. A chain carbon never votes on its own leaflet,
so interdigitation stops mattering and the groups come out equal by
construction:

```bash
chiroflux leaflet-index -topology gromacs_input/topol.tpr \
                        -coords   gromacs_input/conf.g96 \
                        -resname DOPC,POPC -out-dir gromacs_input
```

`-resname` takes several species at once (comma- or space-separated), writing
`CN_<RESNAME>.ndx` per species; `-out` overrides those names, one path per
species in the same order. Doing them together matters: the midplane is then
computed once over every headgroup rather than per species, which on the
reference system is 38.77 A pooled against 38.69 A from DOPC alone and 39.16 A
from the 22 POPC alone — one bilayer should not be cut in two places.

It prints a per-marker count table; every row should match the headgroup split.
On the reference system it reproduces `gmx select` **exactly** for the fifteen
groups that were already right, and changes only the four chain-carbon groups,
by exactly the one and four atoms that were misassigned. Regenerate both
simulations' files and re-run `generate-cvs`.

The old shell recipe also hardcodes `ZMID=4.06` nm where the actual mean
headgroup plane is 3.87 nm; `leaflet-index` computes the midplane from the
structure, and takes `-midplane` if you need to pin it.

### Comparing the lipid preference of two runs

`preference-compare` differences `frac_DOPC(OP_Lamb)` between two simulations.
Two things make this less obvious than it looks.

**The enrichment curves cannot be differenced directly.** Since
`frac_POPC = 1 - frac_DOPC` and `E = frac/F`, the two are locked together as
`E_POPC = 6 - 5*E_DOPC` — the same number drawn twice, on lever arms that
differ fivefold. E_DOPC lives in [0, 1.20] and E_POPC in [0, 6.00], so one
observation ("DOPC takes 66% of contacts where its abundance predicts 83%")
reads as *E_DOPC = 0.795, just under 1* and *E_POPC = 2.02, doubly enriched*.
Differencing either inherits that distortion. `frac_DOPC` is the one free
quantity and the only undistorted scale, so that is what is compared.

**A leaflet a run never touches does not read as blank.** A permeant entering
from below has empty `*_u_*` columns; all their weight sits in the lowest CN
bin, the first moment collapses to (bin centre 0.05) x (frame weight), the
frame weights cancel between the species, and `frac_DOPC` comes out at exactly
0.5 in every bin — a clean flat line at `E_POPC = 3.0`. Pairing two runs by
like-named label therefore compares real data against a fabrication, and
differencing two empty columns gives exactly zero, which reads as perfect
agreement.

`-leaflet-l`/`-leaflet-d` declare which leaflet each run contacts, and pairs
are matched on the contact-type name, so L's `CA_C2_l_DOPC` is compared with
D's `CA_C2_u_DOPC` under the name `CA_C2`. The name drops the species as well
as the leaflet: every pair *is* a DOPC-against-POPC comparison, so a trailing
`_DOPC` in a file name or a row label would suggest a DOPC-only quantity.
Degenerate bins are masked rather than counted as agreement. Both runs must
have been through `chiroflux histograms` first, since this reads their
per-chunk `intermediates/`.

There are 17 comparisons: 16 marker-atom contact types, plus `whole_lipid`
from the plain `DOPC`/`POPC` columns (every atom of a species within 5 A of
the permeant, over that species' atoms per lipid). That last one is the most
direct preference measure of the set, since it privileges no marker atom, and
it carries no leaflet split — none is needed, because a permeant that only
reaches one leaflet makes the whole-system count the near-leaflet count.

Note that the *signed* chirality CVs (`PRO_hCN_*`, `PRO_nCos_*`) cannot join
this table: an enrichment ratio needs a non-negative contact count and those
are 40-43% negative, so `frac = D/(D+P)` is undefined for them. `PRO_rMin_*`
is excluded for a different reason — it is a distance, not a count. Those
belong in `shap-enantiomer`.

```bash
chiroflux preference-compare \
  -dir-l L/analysis_output -dir-d D/analysis_output \
  -label-l L -label-d D \
  -leaflet-l lower -leaflet-d upper \
  -paths reactive -ensemble plus
```

The confidence interval comes from resampling chunks in *both* runs inside one
loop, so it is an interval on the difference rather than two independent
intervals combined after the fact. Measured false-positive rate under a true
null is 5.3% at 40 chunks against a nominal 5%, rising to 8.3% at 10 chunks —
the usual mild over-rejection of a percentile bootstrap with few resampling
units, worth remembering for a run with few chunks.

### `PRO_ang_OH` measures the wrong angle

The index files came from a hand-maintained `make_ndx.py` copied into each
simulation directory, and the copies drifted. For `PRO_C_O_H.ndx` — the group
behind `PRO_ang_OH` — the copies disagree:

| copy | selection |
| --- | --- |
| `D_PRO_neutral/infinit_entry` | `C, O01, H02` |
| `D_PRO_neutral/infinit_escape` | `C, O, H02` |
| `L_PRO_neutral/infinit_entry` | `C, O, H02` |

In the CHARMM topology (identical in both runs) `O` is the carbonyl oxygen,
type OG2D1, bonded only to `C`; `O01` is the hydroxyl, type OG311, bonded to
`C` and `H02`. The carboxylic C–O–H angle is therefore **C–O01–H02**. The
`C, O, H02` spelling asks for the angle at the carbonyl oxygen subtended by a
hydrogen it is not bonded to — a well-defined number, but not the one the CV
name claims, and not an internal coordinate of the molecule.

The shipped `D_PRO_neutral/infinit_escape/gromacs_input/PRO_C_O_H.ndx` holds
indices `[2, 3, 16]` = C, **O**, H02, so the escape run's `PRO_ang_OH` is the
carbonyl version. Check L's shipped file before comparing the two runs; if they
disagree, `PRO_ang_OH` is a convention artefact and will act as a spurious L/D
discriminator.

`chiroflux permeant-index` writes `C, O01, H02` for both runs and verifies it
against the topology's bond list, so **regenerating changes what `PRO_ang_OH`
means**. That is the intended fix, not a regression. If you did want the
carbonyl geometry, change the one entry in `PERMEANT_INDEX_SPEC` — but then
rename the CV, because `dih_OH` next to it is already the proper carboxylic
torsion `O-C-O01-H02`.

### Tied ranks in the weighted Spearman

`_weighted_spearman` in `statistical_analysis.py` does not average tied ranks:
`np.argsort(np.argsort(x))` gives a constant column the distinct ranks
`0..n-1`, so its `std_r < 1e-12` guard can never fire and a degenerate CV
reports a large spurious ρ instead of 0. Covered by an `xfail` test in
`tests/test_weighting.py`. Cohen's d and the KS distance are unaffected.

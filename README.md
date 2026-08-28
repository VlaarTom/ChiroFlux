# ChiroFlux

Collective-variable analysis for TIS/RETIS path sampling with
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

Note that options use a single dash (`-toml`, `-cv-dir`), matching the infretis
tooling convention rather than the GNU `--long-option` style.

### Aligning angle conventions between two simulations

A CV that differs between L and D purely by convention will "perfectly"
separate them while carrying no physical information, and will dominate any
importance ranking. `shap-enantiomer` has two corrections for this. They fix
*different* things, and which you need depends on how the angle is defined:

| flag | operation | corrects for | applies to |
| --- | --- | --- | --- |
| `-mirror-l` / `-mirror-d` | θ → −θ | **chirality**: a chirality-odd CV is negated between mirror-image enantiomers by definition, so L has φ where D has −φ | **signed** CVs on a zero-centred domain, e.g. a dihedral from `arctan2` in [−180, 180] |
| `-flip-l` / `-flip-d` | θ → 180 − θ | **direction of entry/internal/escape**: a permeant entering from the other leaflet sees the membrane normal reversed, swapping the +z (extracellular) and −z (intracellular) face | **unsigned** angles against the membrane normal, e.g. anything from `arccos` in [0, 180] |

`-flip-*` is not a chirality operation — it will not turn L into D. 90°
is its pivot because that is the flat-vs-vertical boundary: below 90° the
reference face points toward +z, above it toward −z.

Match the flag to the domain. Negating an `arccos` angle sends [0, 180] to
[−180, 0], off its own domain; entry-flipping a signed dihedral gives
180 − (−170) = 350, likewise off-domain. Both operations are their own
inverse, both flip the sign of the cosine, and both should be applied to one
simulation only, so its values become comparable to the other's.

Columns already folded to `cos(...)`/`cos2(...)` by `-angle-cols` are skipped
by `-flip-*` with a warning, since they hold values in [−1, 1] rather
than degrees. (Negating them is the equivalent operation there, because
−cos θ = cos(180 − θ).)

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

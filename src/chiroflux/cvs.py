"""Transformations applied to a CV matrix once it has been read.

These operate purely on in-memory arrays of collective-variable values and
their column names - no file access, no simulation concepts. They cover the
adjustments needed before CVs from different sources are comparable:

* angle columns in degrees folded to cos/cos2 so that periodicity does not
  make two identical configurations look far apart;
* z-coordinates re-referenced to a membrane centre;
* per-simulation enantiomer mirrors (theta -> -theta), opposite-leaflet entry
  flips (theta -> 180 - theta) and renames, used to put two simulations into
  one common convention;
* frame subsampling, with a matching count-only version used to size output
  arrays before the data is read.

Depends only on numpy.
"""

import warnings

import numpy as np


def _apply_angle_transforms(cv_array, cv_names, cos_cols=None, cos2_cols=None):
    """Convert angular CV columns from degrees to cos(θ) or cos²(θ).

    Operates on a copy of cv_array so the original is not mutated.

    Parameters
    ----------
    cv_array  : (N_paths, N_cvs, N_interfaces) float array from
                _extract_cv_crossings.
    cv_names  : list of N_cvs column name strings.
    cos_cols  : CV names to transform as cos(θ).  Use for asymmetric
                molecules where θ = 20° and θ = 160° are distinct.
    cos2_cols : CV names to transform as cos²(θ).  Use for head-tail
                symmetric molecules where θ and 180°−θ are equivalent;
                cos²(θ) maps both to the same value.

    Returns
    -------
    cv_array  : transformed copy, same shape.
    cv_names  : updated name list; transformed columns are renamed
                "cos(<name>)" or "cos2(<name>)".
    """
    if not cos_cols and not cos2_cols:
        return cv_array, cv_names

    cv_array = cv_array.copy()
    new_names = list(cv_names)
    name_to_idx = {name: i for i, name in enumerate(cv_names)}

    for col in (cos_cols or []):
        if col not in name_to_idx:
            warnings.warn(
                f"-angle-cols: '{col}' not found in CV names {cv_names}; skipping.",
                stacklevel=2,
            )
            continue
        idx = name_to_idx[col]
        cv_array[:, idx, :] = np.cos(np.deg2rad(cv_array[:, idx, :]))
        new_names[idx] = f"cos({col})"
        print(f"  Transformed '{col}' → cos({col})  [asymmetric angle]")

    for col in (cos2_cols or []):
        if col not in name_to_idx:
            warnings.warn(
                f"-sym-angle-cols: '{col}' not found in CV names {cv_names}; skipping.",
                stacklevel=2,
            )
            continue
        idx = name_to_idx[col]
        cv_array[:, idx, :] = np.cos(np.deg2rad(cv_array[:, idx, :])) ** 2
        new_names[idx] = f"cos2({col})"
        print(f"  Transformed '{col}' → cos²({col})  [symmetric angle]")

    return cv_array, new_names

def _apply_z_corrections(cv_array, cv_names, z_cols=None, z_ref="z_Memb", drop_ref=False):
    """Shift z-coordinate CVs so that the membrane shift is accounted for and
    the value is relative to the membrane midplane (z=0).

    Operates on a copy of cv_array so the original is not mutated.

    Parameters
    ----------
    cv_array : (N_paths, N_cvs, N_interfaces) float array from
               _extract_cv_crossings.
    cv_names : list of N_cvs column name strings.
    z_cols   : CV names to shift by the reference value.
    z_ref    : CV name to use as the reference midplane position.
    drop_ref : If True, remove the z_ref column from cv_array and cv_names
               after the corrections are applied.  Use this when z_ref (e.g.
               z_Memb) carries no independent information and should not appear
               as a feature in the model.

    Returns
    -------
    cv_array : transformed copy, same shape (or one fewer CV column if drop_ref).
    cv_names : name list, unchanged (or with z_ref removed if drop_ref).
    """
    if not z_cols:
        return cv_array, cv_names

    if z_ref not in cv_names:
        raise ValueError(
            f"-z-ref '{z_ref}' not found in CV names {cv_names}; cannot apply "
            "z-coordinate corrections."
        )
    ref_idx = cv_names.index(z_ref)

    cv_array = cv_array.copy()
    for col in z_cols:
        if col not in cv_names:
            warnings.warn(
                f"-z-cols: '{col}' not found in CV names {cv_names}; skipping.",
                stacklevel=2,
            )
            continue
        idx = cv_names.index(col)
        cv_array[:, idx, :] -= cv_array[:, ref_idx, :]

    if drop_ref:
        ref_idx = cv_names.index(z_ref)
        cv_names = [n for n in cv_names if n != z_ref]
        cv_array = np.delete(cv_array, ref_idx, axis=1)

    return cv_array, cv_names

def _apply_cv_mirror(cv_array, cv_names, mirror_substrings):
    """Mirror signed CV columns: theta -> -theta.

    This is the enantiomer mirror operation. A chirality-odd (pseudo-scalar)
    CV such as a signed dihedral is guaranteed by symmetry to come out negated
    between mirror-image enantiomers - L has phi where D has -phi - for purely
    definitional reasons. Left uncorrected it "perfectly" separates the two
    simulations while carrying no information about differential behaviour,
    and dominates any importance ranking. Apply it to one simulation only.

    Only meaningful for CVs on a signed, zero-centred domain such as
    [-180, 180]. An unsigned angle from ``arccos`` lives on [0, 180], and
    negating it would move it to [-180, 0], off its own domain - use
    ``_apply_cv_entry_flip`` for those instead.

    Operates on a copy of cv_array so the original is not mutated.

    Parameters
    ----------
    cv_array          : (N_paths, N_cvs, N_interfaces) float array.
    cv_names          : list of CV column name strings.
    mirror_substrings : list of substrings; any CV whose name contains one is
                        negated.

    Returns
    -------
    cv_array : copy with selected columns multiplied by -1.
    cv_names : unchanged.
    """
    if not mirror_substrings:
        return cv_array, cv_names
    cv_array = cv_array.copy()
    for i, name in enumerate(cv_names):
        if any(sub in name for sub in mirror_substrings):
            cv_array[:, i, :] *= -1.0
    return cv_array, cv_names

def _apply_cv_flip(cv_array, cv_names, entry_flip_substrings):
    """Correct for entry from the opposite membrane leaflet: theta -> 180 - theta.

    For an angle measured against the membrane normal (the lab z-axis), a
    permeant that entered from the other leaflet sees that normal pointing the
    other way, so cos(theta) changes sign and theta becomes 180 - theta. This
    puts paths that entered from opposite sides into one common frame before
    they are compared.

    Concretely, for the chiral ring-plane angle, theta < 90 means the
    substituent-defined face points toward +z (extracellular) and theta > 90
    means it points toward -z (intracellular); this operation swaps the two.
    90 degrees is the flat-vs-vertical boundary, which is why it is the pivot.

    Geometrically it is a reflection about 90 degrees (a reflection about c is
    ``x -> 2c - x``, hence the 180), so it is its own inverse and maps [0, 180]
    back onto [0, 180]. It is *not* a chirality operation - it does not turn L
    into D. For that, see ``_apply_cv_mirror``, which negates signed
    chirality-odd CVs. Do not apply this one to a signed dihedral:
    180 - (-170) = 350 leaves the domain.

    Operates on a copy of cv_array so the original is not mutated.

    Parameters
    ----------
    cv_array              : (N_paths, N_cvs, N_interfaces) float array, degrees.
    cv_names              : list of CV column name strings.
    entry_flip_substrings : list of substrings; any CV whose name contains one
                            is flipped. Columns already converted to cos/cos2
                            by ``_apply_angle_transforms`` are skipped with a
                            warning: they hold values in [-1, 1], not degrees,
                            so flipping them would give nonsense near 179-181.

    Returns
    -------
    cv_array : copy with the selected columns mapped to 180 - theta.
    cv_names : unchanged.
    """
    if not entry_flip_substrings:
        return cv_array, cv_names
    cv_array = cv_array.copy()
    for i, name in enumerate(cv_names):
        if not any(sub in name for sub in entry_flip_substrings):
            continue
        if name.startswith(("cos(", "cos2(")):
            warnings.warn(
                f"'{name}' is already in cosine form, not degrees; skipping the "
                f"entry flip. Negating it is the equivalent operation there, "
                f"since -cos(theta) == cos(180 - theta).",
                stacklevel=2,
            )
            continue
        cv_array[:, i, :] = 180.0 - cv_array[:, i, :]
    return cv_array, cv_names

def _apply_cv_rename(cv_names, rename_pairs):
    """Apply substring substitutions to a list of CV names.

    rename_pairs is a list of (old, new) tuples applied in order.
    Substitutions that do not match any name are silently skipped.
    """
    result = list(cv_names)
    for old, new in rename_pairs:
        result = [name.replace(old, new) for name in result]
    return result

def _subsample_frames(op_vals, cvs, stride, max_frames):
    """Stride and/or cap the frames kept from a single trajectory."""
    if stride > 1:
        op_vals = op_vals[::stride]
        cvs = cvs[::stride]
    if max_frames is not None and len(op_vals) > max_frames:
        idx = np.linspace(0, len(op_vals) - 1, max_frames).round().astype(int)
        op_vals = op_vals[idx]
        cvs = cvs[idx]
    return op_vals, cvs

def _frame_count_after_subsample(n_frames, stride, max_frames):
    """Frame count _subsample_frames would produce for a path with n_frames
    rows, without touching the actual data (mirrors its stride/cap logic)."""
    if stride > 1:
        n_frames = (n_frames + stride - 1) // stride  # len(arr[::stride])
    if max_frames is not None and n_frames > max_frames:
        n_frames = max_frames
    return n_frames

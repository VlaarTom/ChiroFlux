"""Comparing two SASA profiles (e.g. L vs D).

The difference is bootstrapped jointly rather than by drawing two separate
confidence bands and looking for overlap: overlapping intervals do not imply
the absence of a difference, so that reading understates significance.
"""

import json
import warnings

import numpy as np
import pytest
import typer

from chiroflux.sasa_compare import (
    _bootstrap_difference,
    _check_comparable,
    _load_side,
    _significant,
    _weighted_means,
)

NZ = 25


def _side(rng, n_paths, shift=0.0):
    """Synthetic per-path arrays in the shape sasa's intermediates use."""
    w = rng.uniform(5, 20, (n_paths, NZ))
    base = 100 + 30 * np.exp(-((np.arange(NZ) - 12) / 5) ** 2) + shift
    per_path = base + rng.normal(0, 4, (n_paths, NZ))
    return {
        "w": w, "tot": per_path * w, "pol": 0.4 * per_path * w,
        "apo": 0.6 * per_path * w, "free": (per_path + 80) * w,
        "tot2": (per_path ** 2) * w,
    }


class TestWeightedMeans:
    def test_point_estimate_is_the_weight_ratio(self):
        data = {
            "w": np.array([[2.0, 4.0]]),
            "tot": np.array([[20.0, 40.0]]),
            "pol": np.array([[8.0, 16.0]]),
            "apo": np.array([[12.0, 24.0]]),
            "free": np.array([[40.0, 80.0]]),
        }
        means, w = _weighted_means(data)
        assert means["tot"] == pytest.approx([10.0, 10.0])
        assert means["exposure"] == pytest.approx([0.5, 0.5])
        assert w == pytest.approx([2.0, 4.0])

    def test_zero_weight_bins_become_nan_not_a_crash(self):
        data = {k: np.array([[1.0, 0.0]]) for k in ("tot", "pol", "apo", "free")}
        data["w"] = np.array([[1.0, 0.0]])
        means, _ = _weighted_means(data)
        assert np.isnan(means["tot"][1])


class TestBootstrapDifference:
    def test_recovers_a_known_difference(self):
        rng = np.random.default_rng(7)
        res = _bootstrap_difference(_side(rng, 60), _side(rng, 55, shift=8.0),
                                    n_bootstrap=400, alpha=0.05)
        assert np.nanmean(res["delta_tot"]) == pytest.approx(8.0, abs=1.5)

    def test_the_interval_covers_the_truth(self):
        rng = np.random.default_rng(7)
        res = _bootstrap_difference(_side(rng, 60), _side(rng, 55, shift=8.0),
                                    n_bootstrap=400, alpha=0.05)
        covered = np.sum((res["lo_tot"] <= 8.0) & (res["hi_tot"] >= 8.0))
        assert covered >= 0.8 * NZ, f"only {covered}/{NZ} bins covered the truth"

    def test_a_real_difference_is_detected(self):
        rng = np.random.default_rng(7)
        res = _bootstrap_difference(_side(rng, 60), _side(rng, 55, shift=8.0),
                                    n_bootstrap=400, alpha=0.05)
        assert _significant(res, "tot").mean() > 0.8

    def test_false_positive_rate_is_near_the_nominal_alpha(self):
        """With no true difference, ~alpha of bins should be flagged. A badly
        constructed bootstrap shows up here as a much larger rate."""
        rates = []
        for trial in range(12):
            rng = np.random.default_rng(2000 + trial)
            res = _bootstrap_difference(_side(rng, 50), _side(rng, 50),
                                        n_bootstrap=300, alpha=0.05)
            rates.append(_significant(res, "tot").mean())
        assert np.mean(rates) < 0.20, f"false-positive rate {np.mean(rates):.1%}"

    def test_direction_is_b_minus_a(self):
        rng = np.random.default_rng(1)
        res = _bootstrap_difference(_side(rng, 40), _side(rng, 40, shift=10.0),
                                    n_bootstrap=200, alpha=0.05)
        assert np.nanmean(res["delta_tot"]) > 0

    def test_too_few_paths_gives_nan_intervals_not_an_error(self):
        rng = np.random.default_rng(0)
        res = _bootstrap_difference(_side(rng, 1), _side(rng, 1),
                                    n_bootstrap=100, alpha=0.05)
        assert np.isnan(res["lo_tot"]).all()

    def test_low_weight_bins_are_masked_from_both_sides(self):
        rng = np.random.default_rng(0)
        a, b = _side(rng, 30), _side(rng, 30)
        a["w"][:, 0] = 0.0
        res = _bootstrap_difference(a, b, n_bootstrap=100, alpha=0.05)
        assert res["mask"][0]
        assert np.isnan(res["delta_tot"][0])


class TestComparabilityGuards:
    def _meta(self, tmp_path, name, **over):
        d = tmp_path / name
        (d / "intermediates").mkdir(parents=True)
        meta = {"z_range": [-40.0, 40.0], "z_bin_width": 1.0,
                "fold_symmetric": False, "probe_radius": 1.4,
                "occlude_with_water": False}
        meta.update(over)
        (d / "sasa_meta.json").write_text(json.dumps(meta))
        return d

    def test_different_bin_counts_are_rejected(self, tmp_path):
        a = {"w": np.zeros((3, 10))}
        b = {"w": np.zeros((3, 20))}
        with pytest.raises(typer.BadParameter, match="z bins"):
            _check_comparable(str(tmp_path), str(tmp_path), a, b)

    def test_mismatched_binning_metadata_is_rejected(self, tmp_path):
        da = self._meta(tmp_path, "a")
        db = self._meta(tmp_path, "b", z_bin_width=2.0)
        same = {"w": np.zeros((3, 10))}
        with pytest.raises(typer.BadParameter, match="z_bin_width"):
            _check_comparable(str(da), str(db), same, same)

    def test_mismatched_probe_radius_is_rejected(self, tmp_path):
        da = self._meta(tmp_path, "a")
        db = self._meta(tmp_path, "b", probe_radius=1.8)
        same = {"w": np.zeros((3, 10))}
        with pytest.raises(typer.BadParameter, match="probe_radius"):
            _check_comparable(str(da), str(db), same, same)

    def test_matching_metadata_passes(self, tmp_path):
        da = self._meta(tmp_path, "a")
        db = self._meta(tmp_path, "b")
        same = {"w": np.zeros((3, 10))}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_comparable(str(da), str(db), same, same)

    def test_missing_metadata_warns_rather_than_silently_passing(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        same = {"w": np.zeros((3, 10))}
        with pytest.warns(UserWarning, match="sasa_meta.json"):
            _check_comparable(str(tmp_path / "a"), str(tmp_path / "b"), same, same)

    def test_a_directory_without_intermediates_is_rejected(self, tmp_path):
        with pytest.raises(typer.BadParameter, match="intermediates"):
            _load_side(str(tmp_path), "a")


class TestMirrorB:
    """L and D can traverse the membrane in opposite directions, so one side's
    depth axis runs the other way and must be reflected before differencing."""

    @staticmethod
    def _peaked(rng, n_paths, peak, z):
        w = rng.uniform(5, 20, (n_paths, len(z)))
        per = 100 + 40 * np.exp(-((z - peak) / 6) ** 2) + rng.normal(0, 3, (n_paths, len(z)))
        return {
            "w": w, "tot": per * w, "pol": 0.4 * per * w, "apo": 0.6 * per * w,
            "free": (per + 80) * w, "tot2": (per ** 2) * w,
            "zp_up": np.full(n_paths, 18.0), "zp_lo": np.full(n_paths, -20.0),
            "hist2d": np.arange(3 * len(z), dtype=float).reshape(3, len(z)),
        }

    def test_mirroring_aligns_opposite_direction_profiles(self):
        from chiroflux.sasa_compare import _mirror_z

        z = np.linspace(-40, 40, 41)[:-1] + 1.0
        rng = np.random.default_rng(0)
        a = self._peaked(rng, 60, -20, z)
        b = self._peaked(rng, 60, +20, z)

        unmirrored = _significant(_bootstrap_difference(a, b, 300, 0.05), "tot").sum()
        mirrored = _significant(
            _bootstrap_difference(a, _mirror_z(b), 300, 0.05), "tot"
        ).sum()
        assert mirrored < unmirrored / 3, (
            f"mirroring should align the profiles: {unmirrored} -> {mirrored} bins"
        )

    def test_mirror_reverses_the_z_axis_of_every_per_path_array(self):
        from chiroflux.sasa_compare import _mirror_z

        rng = np.random.default_rng(0)
        z = np.linspace(-10, 10, 9)[:-1] + 1.25
        data = self._peaked(rng, 5, 4.0, z)
        out = _mirror_z(data)
        for key in ("w", "tot", "pol", "apo", "free", "tot2"):
            assert np.array_equal(out[key], data[key][:, ::-1]), key

    def test_mirror_is_its_own_inverse(self):
        from chiroflux.sasa_compare import _mirror_z

        rng = np.random.default_rng(0)
        z = np.linspace(-10, 10, 9)[:-1] + 1.25
        data = self._peaked(rng, 5, 4.0, z)
        back = _mirror_z(_mirror_z(data))
        assert np.allclose(back["tot"], data["tot"])
        assert np.allclose(back["zp_up"], data["zp_up"])

    def test_phosphate_planes_swap_and_negate(self):
        """Reflecting turns the upper leaflet into the lower one."""
        from chiroflux.sasa_compare import _mirror_z

        rng = np.random.default_rng(0)
        z = np.linspace(-10, 10, 9)[:-1] + 1.25
        data = self._peaked(rng, 4, 0.0, z)
        out = _mirror_z(data)
        assert np.allclose(out["zp_up"], -data["zp_lo"])
        assert np.allclose(out["zp_lo"], -data["zp_up"])

    def test_hist2d_reflects_only_along_z(self):
        """hist2d is (n_sasa, n_z); the SASA axis must not be touched."""
        from chiroflux.sasa_compare import _mirror_z

        rng = np.random.default_rng(0)
        z = np.linspace(-10, 10, 9)[:-1] + 1.25
        data = self._peaked(rng, 4, 0.0, z)
        out = _mirror_z(data)
        assert np.array_equal(out["hist2d"], data["hist2d"][:, ::-1])

    def test_asymmetric_z_range_is_refused(self):
        """Reversing bins equals z -> -z only on a range centred on zero;
        on an asymmetric one it would shift rather than reflect."""
        from chiroflux.sasa_compare import _require_symmetric_z

        with pytest.raises(typer.BadParameter, match="symmetric"):
            _require_symmetric_z((0.0, 40.0))
        _require_symmetric_z((-40.0, 40.0))

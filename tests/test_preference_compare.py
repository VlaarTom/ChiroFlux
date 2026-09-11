"""Comparing two runs' lipid preference has one trap and one scale question.

The trap is the leaflet. A permeant entering from below never contacts the
upper leaflet, so its `*_u_*` columns are empty — and an empty column does not
read as blank, it reads as frac_DOPC = 0.5 in every bin. Pairing two runs by
like-named label therefore compares real data against a fabricated flat line,
and differencing two *empty* columns gives exactly zero, which looks like
perfect agreement. Pairs are matched on the leaflet-free stem instead.

The scale question is which curve to difference. The two enrichments are
locked together as E_POPC = 6 - 5 E_DOPC, so differencing either one inherits
the fivefold asymmetry of the 5:1 stoichiometry. frac_DOPC is the one free
quantity and the only undistorted scale.
"""

import numpy as np
import pytest

from chiroflux.preference_compare import (
    _leaflet_key,
    _pairs_for_leaflet,
    _stem,
    compare_one_pair,
)

CENTERS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55])
N_OP = 5


def _chunks(rows, n_chunks=10, jitter=0.0, seed=0):
    """`n_chunks` chunks whose weight sits in the given {bin: weight} rows."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_chunks):
        c = np.zeros((len(CENTERS), N_OP))
        for j, w in rows.items():
            c[j, :] = w * (1.0 + jitter * rng.normal(size=N_OP))
        out.append((c, None, CENTERS))
    return out


class TestLeafletPairing:
    def test_stem_strips_the_leaflet_and_the_species(self):
        """Both halves of a pair must reduce to the same contact-type name.

        The name reaches the file names and the overview's row labels, so a
        trailing "_DOPC" there would read as a DOPC-only quantity when the
        plot is a DOPC-vs-POPC comparison.
        """
        assert _stem("CA_C2_l_DOPC") == "CA_C2"
        assert _stem("CA_C2_u_DOPC") == "CA_C2"
        assert _stem("CA_C2_u_POPC") == "CA_C2"

    def test_the_whole_lipid_columns_get_a_name_of_their_own(self):
        """They are called plainly DOPC/POPC, so there is nothing to strip."""
        assert _stem("DOPC") == "whole_lipid"
        assert _stem("POPC") == "whole_lipid"

    def test_the_whole_lipid_pair_appears_on_both_leaflets(self):
        """It has no leaflet split and needs none: a permeant reaching only
        one leaflet makes the whole-system count the near-leaflet count."""
        for leaflet in ("lower", "upper"):
            assert "whole_lipid" in _pairs_for_leaflet(leaflet)

    def test_the_two_leaflets_map_onto_the_same_stems(self):
        """This is what lets L's lower columns be paired with D's upper ones."""
        lower = _pairs_for_leaflet("lower")
        upper = _pairs_for_leaflet("upper")
        assert lower and upper
        assert set(lower) == set(upper)

    def test_each_leaflet_selects_only_its_own_columns(self):
        for leaflet, marker in (("lower", "_l_"), ("upper", "_u_")):
            for name, (col_d, col_p) in _pairs_for_leaflet(leaflet).items():
                if name == "whole_lipid":
                    continue          # leaflet-free by construction
                assert marker in col_d and marker in col_p

    def test_a_bad_leaflet_name_is_refused(self):
        with pytest.raises(ValueError, match="lower.*upper"):
            _leaflet_key("top")


class TestCompareOnePair:
    def _run(self, rows_l, rows_d, n_bootstrap=200, jitter=0.05):
        lam = np.arange(N_OP, dtype=float)
        l_data = (_chunks(rows_l[0], jitter=jitter, seed=1),
                  _chunks(rows_l[1], jitter=jitter, seed=2))
        d_data = (_chunks(rows_d[0], jitter=jitter, seed=3),
                  _chunks(rows_d[1], jitter=jitter, seed=4))
        return compare_one_pair(l_data, d_data, lam, n_bootstrap, 0.05,
                                np.random.default_rng(0))

    def test_identical_runs_give_a_zero_difference(self):
        rows = ({4: 100.0}, {1: 100.0})
        res = self._run(rows, rows, jitter=0.0)
        assert res["diff"] == pytest.approx(0.0, abs=1e-12)
        assert not res["significant"].any()

    def test_the_interval_is_real_and_contains_the_observation(self):
        """Two runs drawn from the same distribution but different noise.

        A percentile bootstrap must produce a non-degenerate interval that
        brackets the point estimate; a collapsed band would be the symptom of
        resampling something that cannot move.

        Deliberately no "nothing is significant" assertion here: over 60
        true-null replicates the measured false-positive rate is 5.3% at 40
        chunks (nominal 5%) and 8.3% at 10, so a single 5-bin realization
        flagging one bin is ordinary, not a defect. The mild over-rejection at
        small chunk counts is the known behaviour of a percentile bootstrap
        with few resampling units — worth remembering when a run has few
        chunks, but not something a unit test can assert on one draw.
        """
        rows = ({4: 100.0}, {1: 100.0})
        res = self._run(rows, rows)
        assert np.all(res["ci_hi"] > res["ci_lo"])
        assert np.all(res["ci_lo"] <= res["diff"] + 1e-9)
        assert np.all(res["diff"] <= res["ci_hi"] + 1e-9)

    def test_a_planted_difference_is_recovered_with_the_right_sign(self):
        # Rows are CN *bins*; CENTERS gives the contact number each stands for.
        f = lambda d, p: CENTERS[d] / (CENTERS[d] + CENTERS[p])  # noqa: E731
        res = self._run(({4: 100.0}, {1: 100.0}),
                        ({2: 100.0}, {1: 100.0}), jitter=0.0)
        assert res["diff"] == pytest.approx(f(4, 1) - f(2, 1), abs=1e-9)
        assert np.all(res["diff"] > 0)

    def test_a_large_planted_difference_is_significant(self):
        res = self._run(({5: 100.0}, {1: 100.0}),
                        ({1: 100.0}, {5: 100.0}))
        assert res["significant"].all()
        assert np.all(res["p_value"] <= 0.05)

    def test_an_empty_leaflet_is_masked_rather_than_compared(self):
        """Both runs' weight in the lowest CN bin: no contacts to apportion.

        Differencing them would give exactly 0.0 and read as agreement.
        """
        rows = ({0: 100.0}, {0: 100.0})
        res = self._run(rows, rows, jitter=0.0)
        assert res["mask"].all()
        assert np.isnan(res["diff"]).all()
        assert not res["significant"].any()

    def test_one_empty_run_masks_the_comparison(self):
        """Real data against an empty column is not a difference worth having."""
        res = self._run(({4: 100.0}, {1: 100.0}),
                        ({0: 100.0}, {0: 100.0}), jitter=0.0)
        assert res["mask"].all()
        assert np.isnan(res["diff"]).all()

    def test_chunk_counts_are_reported(self):
        res = self._run(({4: 100.0}, {1: 100.0}), ({4: 100.0}, {1: 100.0}))
        assert res["n_chunks_l"] == 10 and res["n_chunks_d"] == 10

    def test_the_difference_is_on_the_fraction_not_the_enrichment(self):
        """frac is bounded by [0,1]; E_POPC would put the same gap on a 0..6
        scale and E_DOPC on a 0..1.2 one, which is the distortion this avoids."""
        res = self._run(({5: 100.0}, {1: 100.0}),
                        ({1: 100.0}, {5: 100.0}), jitter=0.0)
        assert np.all(np.abs(res["diff"]) <= 1.0)

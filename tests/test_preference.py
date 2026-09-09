"""The DOPC/POPC preference statistics must measure contacts, not frames.

The 2D intermediates histogram (coordination number, OP_Lamb), so the number of
contacts at an OP bin is the *first* moment along the CN axis — weight each CN
bin by the CN it stands for. Summing the raw counts instead gives the frame
weight, and because every frame carries a value in both the DOPC and the POPC
column, the two species then produce the *same number*: frac_DOPC comes out as
exactly 0.5 in every bin whatever the simulation did, enrichment sits at a
constant 0.6, the bootstrap band collapses to nothing, and no bin is ever
significant. That is indistinguishable from "the plot has no data", which is
how it was found.

The cases below have answers that can be worked out by hand, which is the point:
5 DOPC contacts against 1 POPC contact is exactly the 5:1 bulk stoichiometry and
must give an enrichment of exactly 1.
"""

import numpy as np
import pytest

from chiroflux import cv_histograms as cvh
from chiroflux.cv_histograms import (
    F_DOPC,
    F_POPC,
    _compute_enrichment_from_chunks,
    compute_dopc_popc_preference_statistics,
)

CN_CENTERS = np.arange(6.0)
N_OP = 4


def _chunk(cn_value, weight=100.0, n_op=N_OP, spread=None):
    """A chunk whose every frame sits in the `cn_value` coordination bin."""
    counts = np.zeros((len(CN_CENTERS), n_op))
    counts[cn_value, :] = weight if spread is None else spread
    return (counts, None, CN_CENTERS)


def _enrich(cn_dopc, cn_popc, **kw):
    return _compute_enrichment_from_chunks(
        [_chunk(cn_dopc, **kw)], [_chunk(cn_popc, **kw)], np.arange(N_OP)
    )


class TestContactFraction:
    def test_bulk_stoichiometry_gives_an_enrichment_of_exactly_one(self):
        """5 DOPC : 1 POPC contacts is the 5:1 bulk ratio — no preference."""
        r = _enrich(5, 1)
        assert r["frac_dopc"] == pytest.approx(F_DOPC)
        assert r["enrich_dopc"] == pytest.approx(1.0)

    def test_the_fraction_is_not_pinned_at_one_half(self):
        """The regression guard: the zeroth moment gives 0.5 for every input."""
        for cn_d, cn_p in [(5, 1), (4, 2), (1, 3), (2, 2)]:
            frac = _enrich(cn_d, cn_p)["frac_dopc"]
            expected = cn_d / (cn_d + cn_p)
            assert frac == pytest.approx(expected), (cn_d, cn_p)

    def test_equal_contact_numbers_do_mean_a_half(self):
        """0.5 is a legitimate answer — when the contacts really are equal."""
        assert _enrich(3, 3)["frac_dopc"] == pytest.approx(0.5)

    def test_depletion_and_enrichment_have_the_right_sense(self):
        r = _enrich(4, 2)  # less DOPC than the 5:1 ratio would give
        assert np.all(r["enrich_dopc"] < 1.0)
        assert (1.0 - r["frac_dopc"][0]) / F_POPC > 1.0

    def test_the_extremes_are_the_reciprocal_stoichiometries(self):
        assert _enrich(1, 0)["enrich_dopc"] == pytest.approx(1.0 / F_DOPC)
        assert (1.0 - _enrich(0, 1)["frac_dopc"][0]) / F_POPC == pytest.approx(
            1.0 / F_POPC
        )

    def test_contacts_and_frames_are_reported_separately(self):
        """Their ratio is the mean CN; equality means the moment was lost."""
        r = _enrich(5, 1, weight=100.0)
        assert r["frames_dopc"] == pytest.approx(100.0)
        assert r["contacts_dopc"] == pytest.approx(500.0)
        assert r["contacts_popc"] == pytest.approx(100.0)


class TestMasking:
    def test_sparse_bins_are_masked_on_frame_weight(self):
        counts_d = np.zeros((6, N_OP))
        counts_p = np.zeros((6, N_OP))
        counts_d[5, :] = [1000.0, 1000.0, 1.0, 1000.0]  # bin 2 is 0.1% of peak
        counts_p[1, :] = [200.0, 200.0, 0.2, 200.0]
        r = _compute_enrichment_from_chunks(
            [(counts_d, None, CN_CENTERS)], [(counts_p, None, CN_CENTERS)],
            np.arange(N_OP),
        )
        assert r["mask"].tolist() == [False, False, True, False]
        assert np.isnan(r["frac_dopc"][2])

    def test_a_bin_with_no_contacts_is_masked_not_divided_by_zero(self):
        """Well sampled but every frame has CN 0 in both species."""
        r = _enrich(0, 0)
        assert r["mask"].all()
        assert np.isnan(r["frac_dopc"]).all()

    def test_empty_input_masks_everything(self):
        empty = (np.zeros((6, N_OP)), None, CN_CENTERS)
        r = _compute_enrichment_from_chunks([empty], [empty], np.arange(N_OP))
        assert r["mask"].all()

    def test_the_mask_is_boolean(self):
        """It indexes other arrays; a float mask would index by position."""
        assert _enrich(5, 1)["mask"].dtype == np.bool_


class TestBootstrapStatistics:
    """End-to-end through the chunk bootstrap, with the loader stubbed."""

    @pytest.fixture
    def stubbed_chunks(self, monkeypatch):
        rng = np.random.default_rng(0)

        def make(cn_value, n_chunks=12):
            out = []
            for _ in range(n_chunks):
                counts = np.zeros((len(CN_CENTERS), N_OP))
                # Vary the weight per chunk so resampling has something to move.
                counts[cn_value, :] = rng.uniform(80.0, 120.0, size=N_OP)
                out.append((counts, None, CN_CENTERS))
            return out

        def fake_loader(group_key, col):
            return make(5 if col.endswith("DOPC") else 1)

        monkeypatch.setattr(cvh, "load_per_chunk_2d", fake_loader)
        return None

    def test_produces_a_finite_ci_around_the_true_value(self, stubbed_chunks):
        st = compute_dopc_popc_preference_statistics(
            ("reactive", "plus"), "X_DOPC", "X_POPC",
            np.arange(N_OP), n_bootstrap=200,
        )
        assert st is not None
        assert np.all(np.isfinite(st["enrich_dopc"]))
        # 5:1 contacts is exactly bulk, so the CI must bracket 1.0.
        assert np.all(st["ci_lo_dopc"] <= 1.0 + 1e-9)
        assert np.all(st["ci_hi_dopc"] >= 1.0 - 1e-9)

    def test_a_true_null_is_not_called_significant(self, stubbed_chunks):
        """At exactly bulk stoichiometry nothing should come out significant."""
        st = compute_dopc_popc_preference_statistics(
            ("reactive", "plus"), "X_DOPC", "X_POPC",
            np.arange(N_OP), n_bootstrap=200,
        )
        assert not st["significant_dopc"].any()

    def test_a_real_preference_is_detected(self, monkeypatch):
        rng = np.random.default_rng(1)

        def fake_loader(group_key, col):
            cn = 1 if col.endswith("DOPC") else 1  # equal contacts != 5:1 bulk
            out = []
            for _ in range(12):
                counts = np.zeros((len(CN_CENTERS), N_OP))
                counts[cn, :] = rng.uniform(95.0, 105.0, size=N_OP)
                out.append((counts, None, CN_CENTERS))
            return out

        monkeypatch.setattr(cvh, "load_per_chunk_2d", fake_loader)
        st = compute_dopc_popc_preference_statistics(
            ("reactive", "plus"), "X_DOPC", "X_POPC",
            np.arange(N_OP), n_bootstrap=200,
        )
        # Equal contact numbers against a 5:1 bulk means DOPC is depleted.
        assert np.all(st["enrich_dopc"] < 1.0)
        assert st["significant_dopc"].all()

    def test_significance_agrees_with_the_plotted_confidence_band(
        self, stubbed_chunks
    ):
        """The band and the marker are drawn on the same axes from the same
        resamples, so a bin whose CI excludes 1 must also be flagged, and one
        whose CI brackets 1 must not. The old p-value broke exactly this: it
        compared |E_boot - 1| against |E_obs - 1| and so sat at ~0.5 for any
        effect size, marking nothing while the band clearly excluded 1.
        """
        st = compute_dopc_popc_preference_statistics(
            ("reactive", "plus"), "X_DOPC", "X_POPC",
            np.arange(N_OP), n_bootstrap=400,
        )
        ci_excludes_one = (st["ci_lo_dopc"] > 1.0) | (st["ci_hi_dopc"] < 1.0)
        assert st["significant_dopc"].tolist() == ci_excludes_one.tolist()

    def test_a_huge_effect_reaches_the_smallest_representable_p(self, monkeypatch):
        """With a tight bootstrap far from the null, p must hit its 1/n floor,
        not the ~0.5 the un-pivoted comparison returned."""
        def fake_loader(group_key, col):
            counts = np.zeros((len(CN_CENTERS), N_OP))
            counts[1 if col.endswith("DOPC") else 5, :] = 100.0
            return [(counts.copy(), None, CN_CENTERS) for _ in range(12)]

        monkeypatch.setattr(cvh, "load_per_chunk_2d", fake_loader)
        st = compute_dopc_popc_preference_statistics(
            ("reactive", "plus"), "X_DOPC", "X_POPC",
            np.arange(N_OP), n_bootstrap=200,
        )
        assert np.all(st["p_value_dopc"] == pytest.approx(1.0 / 200))

    def test_missing_intermediates_return_none_rather_than_crashing(
        self, monkeypatch
    ):
        monkeypatch.setattr(cvh, "load_per_chunk_2d", lambda *a: [])
        assert compute_dopc_popc_preference_statistics(
            ("reactive", "plus"), "X_DOPC", "X_POPC", np.arange(N_OP)
        ) is None


class TestPairTable:
    def test_every_pair_has_a_distinct_label(self):
        """Labels key the stats dict and name the .png, so a duplicate makes
        one pair silently overwrite another."""
        labels = [label for _, _, label in cvh.DOPC_POPC_PAIRS]
        duplicates = {x for x in labels if labels.count(x) > 1}
        assert not duplicates, f"duplicate pair labels: {sorted(duplicates)}"

    def test_each_pair_names_a_dopc_and_a_popc_column(self):
        for col_d, col_p, label in cvh.DOPC_POPC_PAIRS:
            assert col_d.endswith("_DOPC"), label
            assert col_p.endswith("_POPC"), label
            assert col_d[:-5] == col_p[:-5], label

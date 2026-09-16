"""The resistance integral has to be right, so it is checked against cases
whose answer can be written down.

A flat barrier of height G over a width w with constant D contributes exactly
w*exp(G)/D to 1/P, which pins the units as well as the arithmetic: z in A and
D in A^2/ps make 1/P come out in ps/A, and P in A/ps is 1e4 cm/s.

The other thing worth pinning is the staging. A WHAM profile is referenced to
its own state A, so two stages of one membrane are each defined only up to an
additive constant; splicing them has to remove that constant, or the barrier
acquires a step at the junction and the exponential turns the step into a
factor in P.
"""

import numpy as np
import pytest

from chiroflux.permeation_process import (
    ANGSTROM_PER_PS_TO_CM_PER_S,
    Stage,
    combine_stage_histograms,
    find_cv_files,
    free_energy_from_counts,
    free_energy_from_runs,
    join_profiles,
    junction_steps,
    load_free_energy,
    load_stage,
    midplane_from_cv_files,
    mirror_back_pcross,
    permeability_from_resistance,
    read_wham_header,
    resistance_profile,
    stage_factors,
    stage_shifts_from_pcross,
    stitch_runs,
    stitch_stage_histograms,
    write_stitching,
)


def _stage(name, lam_a, lam_b, pcross, n_paths=1.0, counts=None,
           centers=None):
    centers = np.arange(-9.75, 10.0, 0.5) if centers is None else centers
    counts = np.ones_like(centers) if counts is None else counts
    return Stage(name=name, centers=centers, counts=counts, lam_a=lam_a,
                 lam_b=lam_b, pcross=pcross, n_paths=n_paths)


def _write_wham(root, name, lam_a, lam_b, pcross, paths, counts,
                centers=None):
    """A minimal stand-in for one run's wham/ folder."""
    centers = np.arange(-9.75, 10.0, 0.5) if centers is None else centers
    wham = root / name / "wham"
    wham.mkdir(parents=True)
    header = f"# lm1=-35, lA={lam_a}, lB={lam_b}, first_bin_after_lA_index=3\n"
    with (wham / "histo_probability.txt").open("w") as fh:
        fh.write(header)
        for z, c in zip(centers, counts):
            fh.write(f"{z:.6e} {c:.6e}\n")
    (wham / "Pcross.txt").write_text(
        f"#lam\tP-wham\tP-point\tP-wham2\n"
        f"{lam_a} 1 1 1\n{lam_b} {pcross} {pcross} {pcross}\n"
    )
    (wham / "runav_rate.txt").write_text(
        f"#counter rate\n0 1.0\n{paths} 1.0\n"
    )
    return root / name


def _write_cv_file(path, op, z, op_col="OP_Lamb", z_col="z_PRO"):
    """A per-path CV file: three comment lines, a name line, then data."""
    with path.open("w") as fh:
        fh.write("# reactive\n# plus ensemble\n# 0 duplicates removed\n")
        fh.write(f"{op_col} filler {z_col}\n")
        for o, zz in zip(op, z):
            fh.write(f"{o:.6f} 0.0 {zz:.6f}\n")


class TestResistanceIntegral:
    def test_flat_barrier_matches_the_closed_form(self):
        """1/P = w * exp(G) / D for a square barrier."""
        z = np.linspace(0.0, 10.0, 1001)
        g = np.full_like(z, 2.0)
        d = np.full_like(z, 0.5)
        _, inv_p = permeability_from_resistance(z, resistance_profile(z, g, d))
        assert inv_p == pytest.approx(10.0 * np.exp(2.0) / 0.5, rel=1e-9)

    def test_zero_barrier_is_just_width_over_D(self):
        z = np.linspace(0.0, 4.0, 401)
        _, inv_p = permeability_from_resistance(
            z, resistance_profile(z, np.zeros_like(z), np.full_like(z, 2.0))
        )
        assert inv_p == pytest.approx(4.0 / 2.0, rel=1e-9)

    def test_units_convert_to_cm_per_s(self):
        z = np.linspace(0.0, 1.0, 101)
        p, inv_p = permeability_from_resistance(
            z, resistance_profile(z, np.zeros_like(z), np.ones_like(z))
        )
        assert inv_p == pytest.approx(1.0)              # ps/A
        assert p == pytest.approx(ANGSTROM_PER_PS_TO_CM_PER_S)

    def test_the_barrier_dominates_exponentially(self):
        """Raising the barrier by ln(10) must divide P by ~10."""
        z = np.linspace(0.0, 10.0, 1001)
        d = np.ones_like(z)
        p_low, _ = permeability_from_resistance(
            z, resistance_profile(z, np.zeros_like(z), d))
        p_high, _ = permeability_from_resistance(
            z, resistance_profile(z, np.full_like(z, np.log(10.0)), d))
        assert p_low / p_high == pytest.approx(10.0, rel=1e-9)

    def test_a_thin_high_barrier_outweighs_a_wide_low_one(self):
        """Marrink & Berendsen's point: the profile is not an average."""
        z = np.linspace(0.0, 20.0, 2001)
        d = np.ones_like(z)
        wide_low = np.full_like(z, 1.0)
        thin_high = np.where(np.abs(z - 10.0) < 0.5, 8.0, 0.0)
        _, inv_wide = permeability_from_resistance(
            z, resistance_profile(z, wide_low, d))
        _, inv_thin = permeability_from_resistance(
            z, resistance_profile(z, thin_high, d))
        assert inv_thin > inv_wide

    def test_integration_is_direction_independent(self):
        """A profile handed over in descending z must give the same P."""
        z = np.linspace(0.0, 10.0, 501)
        g = 3.0 * np.exp(-((z - 5.0) ** 2))
        d = np.full_like(z, 0.4)
        fwd, _ = permeability_from_resistance(z, resistance_profile(z, g, d))
        rev, _ = permeability_from_resistance(z[::-1],
                                              resistance_profile(z, g, d)[::-1])
        assert fwd == pytest.approx(rev, rel=1e-12)

    def test_a_slower_region_raises_the_resistance(self):
        z = np.linspace(0.0, 10.0, 1001)
        g = np.zeros_like(z)
        fast = np.ones_like(z)
        slow = np.where(np.abs(z - 5.0) < 1.0, 0.01, 1.0)
        _, inv_fast = permeability_from_resistance(z, resistance_profile(z, g, fast))
        _, inv_slow = permeability_from_resistance(z, resistance_profile(z, g, slow))
        assert inv_slow > inv_fast


class TestLoadFreeEnergy:
    def test_unsampled_bins_are_dropped(self, tmp_path):
        """wham writes inf where nothing was sampled; exp(inf) would swamp
        the integral if those were treated as an infinite barrier."""
        p = tmp_path / "fe.txt"
        p.write_text("# header\n-2.0 inf\n-1.0 0.0\n0.0 1.0\n1.0 inf\n")
        z, g = load_free_energy(str(p))
        assert z.tolist() == [-1.0, 0.0]
        assert g.tolist() == [0.0, 1.0]

    def test_rows_come_back_sorted_by_position(self, tmp_path):
        p = tmp_path / "fe.txt"
        p.write_text("1.0 2.0\n-1.0 0.0\n0.0 1.0\n")
        z, _ = load_free_energy(str(p))
        assert np.all(np.diff(z) > 0)

    def test_a_profile_with_nothing_sampled_is_refused(self, tmp_path):
        p = tmp_path / "fe.txt"
        p.write_text("0.0 inf\n1.0 inf\n")
        with pytest.raises(ValueError, match="fewer than two sampled"):
            load_free_energy(str(p))


class TestJoinProfiles:
    def test_the_overlap_fallback_removes_a_constant_offset(self):
        """Without -pcross the stages are matched on their overlap. That is the
        un-referenced fallback, and it recovers a pure constant offset."""
        z1 = np.linspace(0.0, 10.0, 101)
        z2 = np.linspace(8.0, 18.0, 101)
        truth = lambda t: 0.05 * t ** 2  # noqa: E731
        z, g = join_profiles([(z1, truth(z1)), (z2, truth(z2) + 7.3)])
        assert g == pytest.approx(truth(z), abs=1e-9)

    def test_a_single_profile_is_returned_unchanged(self):
        z1 = np.linspace(0.0, 5.0, 51)
        g1 = np.sin(z1)
        z, g = join_profiles([(z1, g1)])
        assert z == pytest.approx(z1) and g == pytest.approx(g1)

    def test_three_stages_chain_correctly_on_the_fallback(self):
        truth = lambda t: 0.3 * t  # noqa: E731
        stages = []
        for lo, hi, off in ((0, 10, 0.0), (8, 18, 2.5), (16, 26, -4.0)):
            t = np.linspace(lo, hi, 101)
            stages.append((t, truth(t) + off))
        z, g = join_profiles(stages)
        assert g == pytest.approx(truth(z), abs=1e-9)
        assert z.max() == pytest.approx(26.0)

    def test_non_overlapping_stages_are_refused(self):
        """Abutting stages cannot be put on a common scale from the profiles
        alone, so guessing an offset would invent the barrier height."""
        a = (np.linspace(0.0, 5.0, 51), np.zeros(51))
        b = (np.linspace(6.0, 11.0, 51), np.zeros(51))
        with pytest.raises(ValueError, match="does not overlap"):
            join_profiles([a, b])

    def test_the_overlap_window_limits_the_match(self):
        z1 = np.linspace(0.0, 10.0, 101)
        z2 = np.linspace(2.0, 12.0, 101)
        truth = lambda t: 0.1 * t  # noqa: E731
        z, g = join_profiles([(z1, truth(z1)), (z2, truth(z2) + 1.0)],
                             overlap=2.0)
        assert g == pytest.approx(truth(z), abs=1e-9)

    def test_an_empty_list_is_refused(self):
        with pytest.raises(ValueError, match="No free-energy profiles"):
            join_profiles([])


class TestCrossingProbabilityReferencing:
    """Staged WHAM profiles are conditional densities.

    Stage k says where the permeant is *given that it started at stage k's own
    state A*. Putting it on stage 1's scale costs the probability of reaching
    it, so the offset is -sum(ln P_tot) over the intervening junctions. Matching
    the stages on their overlap answers a different question and throws that
    factor away - which is the whole barrier the later stage sits behind, and
    since dG enters the integral exponentially, it is the difference between a
    permeability and a number.
    """

    def test_the_shift_is_minus_log_of_the_crossing_probability(self):
        shifts = stage_shifts_from_pcross([1.0e-3])
        assert shifts[0] == 0.0
        assert shifts[1] == pytest.approx(-np.log(1.0e-3))

    def test_shifts_accumulate_along_the_chain(self):
        p1, p2 = 2.6e-3, 2.9e-4
        shifts = stage_shifts_from_pcross([p1, p2])
        assert shifts == pytest.approx(
            [0.0, -np.log(p1), -np.log(p1) - np.log(p2)]
        )

    def test_every_later_stage_sits_higher(self):
        """P_tot < 1, so reaching a later stage always costs free energy."""
        shifts = stage_shifts_from_pcross([0.1, 0.01, 0.5])
        assert all(b > a for a, b in zip(shifts, shifts[1:]))

    def test_a_certain_crossing_costs_nothing(self):
        assert stage_shifts_from_pcross([1.0]) == pytest.approx([0.0, 0.0])

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_an_impossible_probability_is_refused(self, bad):
        with pytest.raises(ValueError, match="must be in"):
            stage_shifts_from_pcross([bad])

    def test_the_supplied_shift_beats_the_overlap(self):
        """The point of the fix: with -pcross the overlap does not decide."""
        z1 = np.linspace(0.0, 10.0, 101)
        z2 = np.linspace(8.0, 18.0, 101)
        flat = lambda t: np.zeros_like(t)  # noqa: E731
        shifts = stage_shifts_from_pcross([1.0e-3])
        z, g = join_profiles([(z1, flat(z1)), (z2, flat(z2))], shifts=shifts)
        # the overlap would have said 0; the crossing probability says otherwise
        assert g[z > 10.0] == pytest.approx(-np.log(1.0e-3))
        assert g[z < 8.0] == pytest.approx(0.0)

    def test_referencing_changes_the_permeability_by_the_expected_factor(self):
        """Dropping the conditional referencing inflates P by exactly P_tot."""
        z1 = np.linspace(0.0, 10.0, 201)
        z2 = np.linspace(8.0, 20.0, 241)
        zeros1, zeros2 = np.zeros_like(z1), np.zeros_like(z2)
        p_tot = 1.0e-3

        z_f, g_f = join_profiles([(z1, zeros1), (z2, zeros2)])
        z_r, g_r = join_profiles([(z1, zeros1), (z2, zeros2)],
                                 shifts=stage_shifts_from_pcross([p_tot]))
        d = np.ones_like(z_f)
        p_fit, _ = permeability_from_resistance(z_f, resistance_profile(z_f, g_f, d))
        p_ref, _ = permeability_from_resistance(z_r, resistance_profile(z_r, g_r, d))
        # only the stage-2 half is raised, so P falls but by less than 1/p_tot
        assert p_ref < p_fit
        assert 1.0 < p_fit / p_ref < 1.0 / p_tot

    def test_a_shift_per_stage_is_required(self):
        z1 = np.linspace(0.0, 10.0, 11)
        z2 = np.linspace(8.0, 18.0, 11)
        with pytest.raises(ValueError, match="one per stage"):
            join_profiles([(z1, np.zeros(11)), (z2, np.zeros(11))], shifts=[0.0])

    def test_referenced_stages_need_no_overlap(self):
        """Crossing probabilities reference them, so abutting stages are fine."""
        a = (np.linspace(0.0, 5.0, 51), np.zeros(51))
        b = (np.linspace(6.0, 11.0, 51), np.zeros(51))
        z, g = join_profiles([a, b], shifts=stage_shifts_from_pcross([0.01]))
        assert z.size == 102
        assert g[z >= 6.0] == pytest.approx(-np.log(0.01))


class TestWhamStageFiles:
    """Everything needed to reference the stages is in their wham/ folders."""

    def test_the_header_gives_the_stage_its_interfaces(self, tmp_path):
        p = tmp_path / "h.txt"
        p.write_text("# lm1=-35, lA=-25.0, lB=-13.5, total_frames=74.36\n0 1\n")
        header = read_wham_header(str(p))
        assert header["lA"] == -25.0
        assert header["lB"] == -13.5
        assert header["total_frames"] == pytest.approx(74.36)

    def test_a_file_without_a_header_gives_nothing(self, tmp_path):
        p = tmp_path / "h.txt"
        p.write_text("0 1\n")
        assert read_wham_header(str(p)) == {}

    def test_a_stage_reads_its_histogram_and_referencing_numbers(self, tmp_path):
        run = _write_wham(tmp_path, "entry", -25.0, -13.5, 2.6e-3, 8595,
                          np.linspace(1.0, 40.0, 40))
        stage = load_stage(str(run))
        assert stage.lam_a == -25.0 and stage.lam_b == -13.5
        assert stage.pcross == pytest.approx(2.6e-3)
        assert stage.n_paths == pytest.approx(8595)
        assert stage.counts.size == 40
        assert stage.name == "entry"

    def test_the_pcross_column_is_selectable(self, tmp_path):
        run = _write_wham(tmp_path, "entry", -25.0, -13.5, 2.6e-3, 10,
                          np.ones(40))
        assert load_stage(str(run), pcross_col=1).pcross == pytest.approx(2.6e-3)

    def test_a_missing_wham_folder_says_so(self, tmp_path):
        (tmp_path / "entry").mkdir()
        with pytest.raises(ValueError, match="does not exist"):
            load_stage(str(tmp_path / "entry"))

    def test_a_header_without_interfaces_is_refused(self, tmp_path):
        wham = tmp_path / "run" / "wham"
        wham.mkdir(parents=True)
        (wham / "histo_probability.txt").write_text("# lm1=-35\n0.0 1.0\n1.0 1.0\n")
        (wham / "Pcross.txt").write_text("0 1 1 1\n")
        with pytest.raises(ValueError, match="lA, lB"):
            load_stage(str(tmp_path / "run"))


class TestStageFactors:
    """The factor that puts a conditional stage on the first stage's scale."""

    def test_it_reproduces_the_escape_correction_of_cv_histograms(self):
        """With the stages labelled A (entry), M (internal) and B (escape),

            factor_M = N_A * (P_A / P_M_min) / N_M
            factor_B = N_A * (P_A / P_M_min) * (P_M / P_B_min) / N_B

        which is what cv_histograms spells with C and D for M and B.
        """
        p_a, p_m, p_b = 2.624258248023157e-3, 2.7752649339303137e-6, 2.86849e-4
        n_a, n_m, n_b = 8595.0, 5276.0, 10252.0
        stages = [
            _stage("entry", -25.0, -13.5, p_a, n_a),
            _stage("internal", -13.5, 13.5, p_m, n_m),
            _stage("escape", 13.5, 25.0, p_b, n_b),
        ]
        factors = stage_factors(stages, path_norm=True)
        # P_M_min is the escape run and P_B_min the internal one, by symmetry.
        assert factors[1] == pytest.approx(n_a * (p_a / p_b) / n_m)
        assert factors[2] == pytest.approx(
            (n_a * (p_a / p_b) * (p_m / p_m)) / n_b
        )

    def test_by_default_only_the_crossing_probabilities_enter(self):
        """Histograms already normalised per path need no N ratio."""
        p_a, p_m, p_b = 2.624258248023157e-3, 2.7752649339303137e-6, 2.86849e-4
        stages = [
            _stage("entry", -25.0, -13.5, p_a, 8595.0),
            _stage("internal", -13.5, 13.5, p_m, 5276.0),
            _stage("escape", 13.5, 25.0, p_b, 10252.0),
        ]
        factors = stage_factors(stages)
        assert factors[1] == pytest.approx(p_a / p_b)
        assert factors[2] == pytest.approx((p_a / p_b) * (p_m / p_m))

    def test_the_first_stage_is_the_reference(self):
        stages = [_stage("a", -25.0, -13.5, 1e-3, 10.0),
                  _stage("b", -13.5, 13.5, 1e-4, 10.0),
                  _stage("c", 13.5, 25.0, 1e-3, 10.0)]
        assert stage_factors(stages)[0] == 1.0

    def test_the_path_counts_are_applied_only_on_request(self):
        stages = [_stage("a", -25.0, -13.5, 1e-3, 100.0),
                  _stage("b", -13.5, 13.5, 1e-4, 1.0),
                  _stage("c", 13.5, 25.0, 1e-3, 7.0)]
        without = stage_factors(stages)
        with_n = stage_factors(stages, path_norm=True)
        assert with_n[1] == pytest.approx(without[1] * 100.0 / 1.0)
        assert with_n[2] == pytest.approx(without[2] * 100.0 / 7.0)

    def test_a_missing_path_count_is_refused_when_it_is_needed(self):
        stages = [_stage("a", -25.0, -13.5, 1e-3, np.nan),
                  _stage("b", -13.5, 13.5, 1e-4, 1.0),
                  _stage("c", 13.5, 25.0, 1e-3, 1.0)]
        with pytest.raises(ValueError, match="path count"):
            stage_factors(stages, path_norm=True)
        assert stage_factors(stages)[1] > 0       # not needed by default

    def test_the_backward_probabilities_can_be_given_directly(self):
        stages = [_stage("a", -25.0, -13.5, 1e-3, 1.0),
                  _stage("b", -13.5, 0.0, 1e-4, 1.0)]
        factors = stage_factors(stages, pcross_back=[0.5])
        assert factors[1] == pytest.approx(1e-3 / 0.5)

    def test_one_backward_probability_per_junction_is_required(self):
        stages = [_stage("a", -25.0, -13.5, 1e-3, 1.0),
                  _stage("b", -13.5, 0.0, 1e-4, 1.0)]
        with pytest.raises(ValueError, match="one per junction"):
            stage_factors(stages, pcross_back=[0.5, 0.5])

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_an_impossible_crossing_probability_is_refused(self, bad):
        stages = [_stage("a", -25.0, -13.5, bad, 1.0),
                  _stage("b", -13.5, 0.0, 1e-4, 1.0)]
        with pytest.raises(ValueError, match="must be in"):
            stage_factors(stages, pcross_back=[0.5])


class TestMirrorBackPcross:
    """The backward escape out of a stage is the mirrored run, by symmetry."""

    def test_the_mirror_of_the_entry_stage_is_the_escape_stage(self):
        stages = [_stage("entry", -25.0, -13.5, 2.6e-3),
                  _stage("internal", -13.5, 13.5, 2.8e-6),
                  _stage("escape", 13.5, 25.0, 2.9e-4)]
        assert mirror_back_pcross(stages) == [2.9e-4, 2.8e-6]

    def test_the_internal_stage_is_its_own_mirror(self):
        stages = [_stage("entry", -25.0, -13.5, 2.6e-3),
                  _stage("internal", -13.5, 13.5, 2.8e-6),
                  _stage("escape", 13.5, 25.0, 2.9e-4)]
        # second junction: reversing the internal stage mirrors onto itself
        assert mirror_back_pcross(stages)[1] == stages[1].pcross

    def test_an_asymmetric_staging_is_refused_rather_than_guessed(self):
        stages = [_stage("a", -25.0, -13.5, 1e-3),
                  _stage("b", -13.5, 13.5, 1e-4)]
        with pytest.raises(ValueError, match="-pcross-back"):
            mirror_back_pcross(stages)


class TestCombineStageHistograms:
    def test_a_later_stage_is_cut_below_its_own_state_A(self):
        centers = np.arange(-9.75, 10.0, 0.5)
        stages = [_stage("a", -10.0, 0.0, 1e-3, counts=np.ones_like(centers)),
                  _stage("b", 0.0, 10.0, 1e-3, counts=np.ones_like(centers))]
        _, total = combine_stage_histograms(stages, [1.0, 1.0])
        assert total[centers < 0.0] == pytest.approx(1.0)
        assert total[centers > 0.0] == pytest.approx(2.0)

    def test_the_first_stage_keeps_the_bulk_below_its_lA(self):
        centers = np.arange(-9.75, 10.0, 0.5)
        stages = [_stage("a", 0.0, 10.0, 1e-3, counts=np.ones_like(centers))]
        _, total = combine_stage_histograms(stages, [1.0])
        assert total == pytest.approx(1.0)

    def test_the_factor_scales_the_counts(self):
        centers = np.arange(-9.75, 10.0, 0.5)
        stages = [_stage("a", -10.0, 0.0, 1e-3, counts=np.ones_like(centers)),
                  _stage("b", 0.0, 10.0, 1e-3, counts=np.ones_like(centers))]
        _, total = combine_stage_histograms(stages, [1.0, 4.0])
        assert total[centers > 0.0] == pytest.approx(5.0)

    def test_stages_on_different_grids_are_refused(self):
        a = _stage("a", -10.0, 0.0, 1e-3)
        b = _stage("b", 0.0, 10.0, 1e-3, centers=np.arange(-9.0, 10.0, 0.5))
        with pytest.raises(ValueError, match="different bin grid"):
            combine_stage_histograms([a, b], [1.0, 1.0])

    def test_symmetrising_adds_the_mirror_image(self):
        """It is the reverse-direction histogram of the opposite region."""
        centers = np.arange(-9.75, 10.0, 0.5)
        counts = np.where(centers < 0, 3.0, 0.0)
        stages = [_stage("a", -10.0, 0.0, 1e-3, counts=counts)]
        _, total = combine_stage_histograms(stages, [1.0], symmetrize=True)
        assert total == pytest.approx(3.0)

    def test_an_asymmetric_grid_cannot_be_mirrored(self):
        centers = np.arange(0.0, 10.0, 0.5)
        stages = [_stage("a", 0.0, 10.0, 1e-3, centers=centers,
                         counts=np.ones_like(centers))]
        with pytest.raises(ValueError, match="not symmetric"):
            combine_stage_histograms(stages, [1.0], symmetrize=True)


class TestFreeEnergyFromCounts:
    def test_the_free_energy_is_minus_log_of_the_histogram(self):
        z, g = free_energy_from_counts(np.array([0.0, 1.0]),
                                       np.array([1.0, np.e]))
        assert g == pytest.approx([0.0, -1.0])
        assert z == pytest.approx([0.0, 1.0])

    def test_empty_bins_are_dropped_not_treated_as_infinite_barriers(self):
        z, g = free_energy_from_counts(np.arange(4.0),
                                       np.array([1.0, 0.0, 2.0, 0.0]))
        assert z.tolist() == [0.0, 2.0]
        assert np.isfinite(g).all()

    def test_scaling_the_histogram_only_shifts_the_profile(self):
        counts = np.array([1.0, 5.0, 2.0])
        _, g1 = free_energy_from_counts(np.arange(3.0), counts)
        _, g2 = free_energy_from_counts(np.arange(3.0), 17.0 * counts)
        assert np.ptp(g1 - g2) == pytest.approx(0.0, abs=1e-12)

    def test_a_histogram_with_nothing_in_it_is_refused(self):
        with pytest.raises(ValueError, match="Fewer than two sampled"):
            free_energy_from_counts(np.arange(3.0), np.zeros(3))


class TestMidplaneFromCVFiles:
    """op = +/-(z_lab - midplane), so both the offset and the sign are
    measurable from frames that report the order parameter and z together."""

    def test_the_offset_is_recovered_from_an_opposed_axis(self, tmp_path):
        z_nm = np.linspace(2.0, 5.0, 400)                 # nm
        op = 37.6 - 10.0 * z_nm                           # A, flipped
        _write_cv_file(tmp_path / "1.txt", op, z_nm)
        midplane, flip, info = midplane_from_cv_files([tmp_path / "1.txt"])
        assert midplane == pytest.approx(37.6)
        assert flip is True
        assert info["unit"] == "nm"

    def test_an_aligned_axis_is_detected_too(self, tmp_path):
        z_nm = np.linspace(2.0, 5.0, 400)
        op = 10.0 * z_nm - 37.6
        _write_cv_file(tmp_path / "1.txt", op, z_nm)
        midplane, flip, _ = midplane_from_cv_files([tmp_path / "1.txt"])
        assert midplane == pytest.approx(37.6)
        assert flip is False

    def test_a_z_column_already_in_angstrom_needs_no_scaling(self, tmp_path):
        z_a = np.linspace(20.0, 50.0, 400)
        _write_cv_file(tmp_path / "1.txt", 37.6 - z_a, z_a)
        midplane, flip, info = midplane_from_cv_files([tmp_path / "1.txt"])
        assert midplane == pytest.approx(37.6)
        assert info["unit"] == "A"

    def test_noise_averages_out_and_is_reported(self, tmp_path):
        rng = np.random.default_rng(0)
        z_nm = np.linspace(2.0, 5.0, 4000)
        op = 37.6 - 10.0 * z_nm + rng.normal(scale=0.5, size=z_nm.size)
        _write_cv_file(tmp_path / "1.txt", op, z_nm)
        midplane, _, info = midplane_from_cv_files([tmp_path / "1.txt"])
        assert midplane == pytest.approx(37.6, abs=0.05)
        assert info["spread"] == pytest.approx(0.5, rel=0.2)

    def test_several_files_are_pooled(self, tmp_path):
        for i, offset in enumerate((37.0, 38.0)):
            z_nm = np.linspace(2.0, 5.0, 200)
            _write_cv_file(tmp_path / f"{i}.txt", offset - 10.0 * z_nm, z_nm)
        paths = sorted(tmp_path.glob("*.txt"))
        midplane, _, info = midplane_from_cv_files(paths)
        assert midplane == pytest.approx(37.5)
        assert info["n_files"] == 2 and info["n_frames"] == 400

    def test_a_column_that_is_not_a_position_is_refused(self, tmp_path):
        """A slope of neither ~1 nor ~0.1 means the wrong column."""
        z = np.linspace(0.0, 0.001, 200)
        _write_cv_file(tmp_path / "1.txt", np.linspace(-10.0, 10.0, 200), z)
        with pytest.raises(ValueError, match="neither the"):
            midplane_from_cv_files([tmp_path / "1.txt"])

    def test_a_missing_column_lists_what_is_there(self, tmp_path):
        _write_cv_file(tmp_path / "1.txt", np.arange(3.0), np.arange(3.0))
        with pytest.raises(ValueError, match="z_PRO"):
            midplane_from_cv_files([tmp_path / "1.txt"], z_col="z_nope")

    def test_a_constant_order_parameter_cannot_locate_anything(self, tmp_path):
        _write_cv_file(tmp_path / "1.txt", np.ones(50), np.linspace(2.0, 5.0, 50))
        with pytest.raises(ValueError, match="does not vary"):
            midplane_from_cv_files([tmp_path / "1.txt"])


class TestFindCVFiles:
    def test_the_sample_is_spread_over_the_folder_not_taken_from_the_front(
        self, tmp_path
    ):
        for i in range(100):
            (tmp_path / f"{i:03d}.txt").write_text("")
        chosen = find_cv_files(tmp_path, 5)
        assert len(chosen) == 5
        assert chosen[0].name == "000.txt"
        assert chosen[-1].name == "099.txt"

    def test_everything_is_returned_when_there_is_little(self, tmp_path):
        for i in range(3):
            (tmp_path / f"{i}.txt").write_text("")
        assert len(find_cv_files(tmp_path, 10)) == 3
        assert len(find_cv_files(tmp_path, 0)) == 3

    def test_an_empty_folder_says_so(self, tmp_path):
        with pytest.raises(ValueError, match="No per-path"):
            find_cv_files(tmp_path, 5)


class TestFreeEnergyFromRuns:
    """The whole chain, from three run folders to one profile."""

    @staticmethod
    def _three_stages(tmp_path):
        centers = np.arange(-24.75, 25.0, 0.5)
        def band(lo, hi):
            return np.where((centers > lo) & (centers < hi), 1.0, 0.0)
        _write_wham(tmp_path, "entry", -20.0, -10.0, 1.0e-2, 100,
                    band(-25.0, -10.0), centers)
        _write_wham(tmp_path, "internal", -10.0, 10.0, 1.0e-4, 200,
                    band(-12.0, 10.0), centers)
        # deliberately spilling back over its own lA, so the cut is visible
        _write_wham(tmp_path, "escape", 10.0, 20.0, 4.0e-2, 400,
                    3.0 * band(5.0, 25.0), centers)
        return [str(tmp_path / n) for n in ("entry", "internal", "escape")]

    def test_the_stages_are_read_referenced_and_joined(self, tmp_path):
        z, g = free_energy_from_runs(self._three_stages(tmp_path),
                                     symmetrize=False)
        assert z.min() == pytest.approx(-24.75)   # the first stage keeps bulk
        assert z.max() == pytest.approx(24.75)
        assert np.isfinite(g).all()
        # at 9.25 the escape stage has spilled back over its own lA; the cut
        # drops it, leaving the internal stage's 1.0 scaled by its own
        # P_entry / P_escape = 1e-2 / 4e-2
        assert np.exp(-g[np.isclose(z, 9.25)][0]) == pytest.approx(0.25)

    def test_the_step_at_a_junction_is_the_log_of_the_factor(self, tmp_path):
        """Flat stages, so the only structure left is the referencing."""
        runs = self._three_stages(tmp_path)
        p_entry, p_escape, n_entry, n_internal = 1.0e-2, 4.0e-2, 100.0, 200.0

        z, g = free_energy_from_runs(runs, symmetrize=False)
        step = g[z > -10.0][0] - g[z < -10.0][-1]
        # the backward escape out of the internal run is its mirror, escape
        assert step == pytest.approx(-np.log(p_entry / p_escape), rel=1e-9)

        z, g = free_energy_from_runs(runs, symmetrize=False, path_norm=True)
        step = g[z > -10.0][0] - g[z < -10.0][-1]
        assert step == pytest.approx(
            -np.log(n_entry * (p_entry / p_escape) / n_internal), rel=1e-9)

    def test_symmetrising_fills_in_the_far_side(self, tmp_path):
        z, _ = free_energy_from_runs(self._three_stages(tmp_path),
                                     symmetrize=True)
        assert z.max() == pytest.approx(24.75)
        assert z.min() == pytest.approx(-24.75)

    def test_one_stage_needs_no_referencing_at_all(self, tmp_path):
        centers = np.arange(-24.75, 25.0, 0.5)
        _write_wham(tmp_path, "solo", -20.0, -10.0, 1.0e-2, 100,
                    np.ones_like(centers), centers)
        z, g = free_energy_from_runs([str(tmp_path / "solo")], symmetrize=False)
        assert g == pytest.approx(0.0)
        assert z.size == centers.size


class TestStitching:
    """The intermediates of the stitched histogram, kept so they can be looked at."""

    @staticmethod
    def _two_stages():
        centers = np.arange(-9.75, 10.0, 0.5)
        a = _stage("a", -10.0, 0.0, 1e-2, counts=np.full_like(centers, 2.0))
        b = _stage("b", 0.0, 10.0, 1e-2, counts=np.full_like(centers, 3.0))
        return centers, [a, b]

    def test_each_part_is_the_stage_as_it_enters_the_sum(self):
        centers, stages = self._two_stages()
        st = stitch_stage_histograms(stages, [1.0, 0.5])
        assert st.parts[0] == pytest.approx(2.0)               # first keeps all
        assert st.parts[1][centers < 0] == pytest.approx(0.0)  # cut below lA
        assert st.parts[1][centers > 0] == pytest.approx(1.5)  # scaled by 0.5
        assert st.forward == pytest.approx(st.parts[0] + st.parts[1])

    def test_the_mirror_is_the_forward_sum_reflected(self):
        _, stages = self._two_stages()
        st = stitch_stage_histograms(stages, [1.0, 0.5], symmetrize=True)
        assert st.mirror == pytest.approx(st.forward[::-1])
        assert st.total == pytest.approx(st.forward + st.forward[::-1])

    def test_without_symmetrising_the_total_is_the_forward_sum(self):
        """The mirror is still computed, to show what -symmetrize would add."""
        _, stages = self._two_stages()
        st = stitch_stage_histograms(stages, [1.0, 0.5], symmetrize=False)
        assert st.mirror is not None
        assert st.total == pytest.approx(st.forward)

    def test_it_agrees_with_combine_stage_histograms(self):
        _, stages = self._two_stages()
        for symmetrize in (False, True):
            st = stitch_stage_histograms(stages, [1.0, 0.5], symmetrize)
            _, total = combine_stage_histograms(stages, [1.0, 0.5], symmetrize)
            assert st.total == pytest.approx(total)

    def test_an_asymmetric_grid_has_no_mirror(self):
        centers = np.arange(0.25, 10.0, 0.5)
        stages = [_stage("a", 0.0, 10.0, 1e-2, centers=centers,
                         counts=np.ones_like(centers))]
        assert stitch_stage_histograms(stages, [1.0]).mirror is None


class TestJunctionSteps:
    @staticmethod
    def _abutting(centers):
        # The first stage is never cut, so like a real entry histogram it has
        # to end at its own lB; otherwise it overlaps the next stage's range.
        return [_stage("a", -10.0, 0.0, 1e-2,
                       counts=np.where(centers < 0.0, 4.0, 0.0)),
                _stage("b", 0.0, 10.0, 1e-2, counts=np.full_like(centers, 4.0))]

    def test_a_step_that_closes_reads_near_zero(self):
        centers = np.arange(-9.75, 10.0, 0.5)
        stages = self._abutting(centers)
        st = stitch_stage_histograms(stages, [1.0, 1.0])
        (step,) = junction_steps(st)
        assert step["stage"] == "b" and step["lam_a"] == 0.0
        assert step["forward"] == pytest.approx(0.0)

    def test_a_step_that_does_not_close_reads_as_its_log_ratio(self):
        centers = np.arange(-9.75, 10.0, 0.5)
        stages = self._abutting(centers)
        st = stitch_stage_histograms(stages, [1.0, 0.1])
        (step,) = junction_steps(st)
        assert step["forward"] == pytest.approx(-np.log(0.1))

    def test_a_gap_at_the_interface_gives_nan_not_a_fake_step(self):
        centers = np.arange(-9.75, 10.0, 0.5)
        a_counts = np.where(centers < -5.0, 1.0, 0.0)       # stops well short
        stages = [_stage("a", -10.0, 0.0, 1e-2, counts=a_counts),
                  _stage("b", 0.0, 10.0, 1e-2, counts=np.ones_like(centers))]
        (step,) = junction_steps(stitch_stage_histograms(stages, [1.0, 1.0]))
        assert np.isnan(step["forward"])


class TestWriteStitching:
    def test_the_plot_and_table_are_written(self, tmp_path):
        _, stages = TestStitching._two_stages()
        st = stitch_stage_histograms(stages, [1.0, 0.5], symmetrize=True)
        steps = write_stitching(st, str(tmp_path), "test", overw=False)
        assert (tmp_path / "stitching.png").stat().st_size > 0
        with open(tmp_path / "stitching.csv") as fh:
            header = fh.readline().strip().split(",")
        assert header == ["z", "stage_a", "stage_b", "forward", "mirror",
                          "total", "beta_dG_forward", "beta_dG_total"]
        table = np.loadtxt(tmp_path / "stitching.csv", delimiter=",", skiprows=1)
        assert table[:, header.index("total")] == pytest.approx(st.total)
        assert len(steps) == 1

    def test_existing_files_are_not_overwritten_without_O(self, tmp_path):
        _, stages = TestStitching._two_stages()
        st = stitch_stage_histograms(stages, [1.0, 0.5], symmetrize=True)
        write_stitching(st, str(tmp_path), "test", overw=False)
        with pytest.raises(ValueError, match="already exists"):
            write_stitching(st, str(tmp_path), "test", overw=False)
        write_stitching(st, str(tmp_path), "test", overw=True)

    def test_stage_names_become_unique_safe_columns(self, tmp_path):
        centers = np.arange(-9.75, 10.0, 0.5)
        stages = [_stage("run 1", -10.0, 0.0, 1e-2, counts=np.ones_like(centers)),
                  _stage("run,1", 0.0, 10.0, 1e-2, counts=np.ones_like(centers))]
        st = stitch_stage_histograms(stages, [1.0, 1.0], symmetrize=True)
        write_stitching(st, str(tmp_path), "test", overw=False)
        with open(tmp_path / "stitching.csv") as fh:
            header = fh.readline().strip().split(",")
        assert header[1:3] == ["stage_run_1", "stage_run_1_2"]

    def test_stitch_runs_matches_free_energy_from_runs(self, tmp_path):
        runs = TestFreeEnergyFromRuns._three_stages(tmp_path)
        st = stitch_runs(runs, symmetrize=True)
        z_a, g_a = free_energy_from_counts(st.centers, st.total)
        z_b, g_b = free_energy_from_runs(runs, symmetrize=True)
        assert z_a == pytest.approx(z_b) and g_a == pytest.approx(g_b)

"""sasa stores per-path sums at weight 1 and applies the WHAM weights when the
intermediates are analysed.

The weights renormalise whenever the simulations are extended. With them baked
into the intermediates - and the 2-D histogram summed per chunk - every path had
to be recomputed. What has to hold now: each path takes its current weight times
its run's scale exactly once; paths from different runs that share a number stay
apart; the result equals what baking the weights in gave; and intermediates are
only reused when they were computed with the same settings.
"""

import os

import numpy as np
import pytest

from chiroflux import sasa as sa

GROUP = ("reactive", "plus")
N_Z, N_S = 8, 5


def _record(rng, run, path_num, n_frames=40):
    """What _process_path_impl returns for one path, at weight 1."""
    z = rng.integers(0, N_Z, size=n_frames)
    s_tot = rng.uniform(10, 100, size=n_frames)

    def binned(values):
        return np.bincount(z, weights=values, minlength=N_Z).astype(float)

    hist2d = np.zeros((N_S, N_Z))
    np.add.at(hist2d, (rng.integers(0, N_S, size=n_frames), z), 1.0)
    return {
        "path_num": path_num, "group": GROUP, "run": run,
        "w": binned(np.ones(n_frames)), "tot": binned(s_tot),
        "pol": binned(0.6 * s_tot), "apo": binned(0.4 * s_tot),
        "free": binned(np.full(n_frames, 120.0)), "tot2": binned(s_tot ** 2),
        "hist2d": hist2d, "zp_up": 18.0 + rng.normal(), "zp_lo": -18.0 + rng.normal(),
    }


def _weights_file(tmp_path, name, weights):
    path = tmp_path / f"{name}_weights.txt"
    path.write_text("# skip=0\n" + "".join(f"{pn} {w}\n" for pn, w in weights.items()))
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(sa, "INTERMEDIATE_DIR", str(tmp_path / "intermediates"))
    return tmp_path


def _set_runs(monkeypatch, tmp_path, runs):
    """runs: {name: (weights dict, scale)}."""
    monkeypatch.setattr(sa, "RUNS", [
        {"name": name, "weights": _weights_file(tmp_path, name, w), "scale": scale,
         "mirror_z": False}
        for name, (w, scale) in runs.items()])


class TestChunkFormat:
    def test_a_chunk_records_run_path_and_version_but_no_weight(self, workdir):
        rng = np.random.default_rng(0)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 1), _record(rng, "escape", 1)]}, 0)
        (fp,) = sa._group_chunk_files(GROUP)
        with np.load(fp) as d:
            assert int(d["format_version"]) == sa.INTERMEDIATE_VERSION
            assert d["run"].tolist() == ["entry", "escape"]
            assert d["path_num"].tolist() == [1, 1]
            assert d["hist2d"].shape == (2, N_S, N_Z)      # per path now
            assert "weight" not in d.files

    def test_the_buffer_is_emptied(self, workdir):
        buffer = {GROUP: [_record(np.random.default_rng(1), "entry", 1)]}
        sa.flush_chunk(buffer, 0)
        assert buffer[GROUP] == []


class TestLoadingWithWeights:
    def test_each_path_takes_its_weight_times_its_runs_scale(self, workdir, monkeypatch):
        rng = np.random.default_rng(2)
        recs = [_record(rng, "entry", 1), _record(rng, "entry", 2)]
        sa.flush_chunk({GROUP: recs}, 0)
        _set_runs(monkeypatch, workdir, {"entry": ({1: 0.2, 2: 0.5}, 3.0)})
        data = sa.load_group(GROUP, sa.run_weights())

        np.testing.assert_allclose(data["weight"], [0.6, 1.5])
        for key in sa._PER_PATH_SUMS:
            np.testing.assert_allclose(data[key],
                                       np.vstack([0.6 * recs[0][key], 1.5 * recs[1][key]]))
        np.testing.assert_allclose(data["hist2d"],
                                   0.6 * recs[0]["hist2d"] + 1.5 * recs[1]["hist2d"],
                                   rtol=1e-6)

    def test_the_same_path_number_in_two_runs_is_two_paths(self, workdir, monkeypatch):
        rng = np.random.default_rng(3)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 7), _record(rng, "escape", 7)]}, 0)
        _set_runs(monkeypatch, workdir, {"entry": ({7: 1.0}, 1.0),
                                         "escape": ({7: 2.0}, 0.5)})
        data = sa.load_group(GROUP, sa.run_weights())
        assert data["run"].tolist() == ["entry", "escape"]
        np.testing.assert_allclose(data["weight"], [1.0, 1.0])

    def test_paths_without_a_positive_weight_or_run_are_left_out(self, workdir, monkeypatch, capsys):
        rng = np.random.default_rng(4)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 1), _record(rng, "entry", 2),
                                _record(rng, "entry", 3), _record(rng, "gone", 1)]}, 0)
        _set_runs(monkeypatch, workdir, {"entry": ({1: 1.0, 2: 0.0}, 1.0)})
        data = sa.load_group(GROUP, sa.run_weights())
        assert data["path_num"].tolist() == [1]
        assert "3 path(s) without a positive weight" in capsys.readouterr().out

    def test_a_path_in_two_chunks_is_counted_once(self, workdir, monkeypatch):
        rng = np.random.default_rng(5)
        rec = _record(rng, "entry", 1)
        sa.flush_chunk({GROUP: [rec]}, 0)
        sa.flush_chunk({GROUP: [rec]}, 1)
        _set_runs(monkeypatch, workdir, {"entry": ({1: 1.0}, 1.0)})
        assert sa.load_group(GROUP, sa.run_weights())["w"].shape[0] == 1

    def test_an_older_pre_weighted_chunk_is_never_mixed_in(self, workdir, monkeypatch):
        rng = np.random.default_rng(6)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 1)]}, 0)
        old = os.path.join(sa.INTERMEDIATE_DIR, f"0001__{sa.group_tag(GROUP)}__sasa.npz")
        np.savez_compressed(old, w=np.ones((1, N_Z)), tot=np.ones((1, N_Z)),
                            path_num=np.array([2]), weight=np.array([1.0]))
        _set_runs(monkeypatch, workdir, {"entry": ({1: 1.0, 2: 1.0}, 1.0)})
        assert sa.load_group(GROUP, sa.run_weights())["path_num"].tolist() == [1]

    def test_the_combined_group_pools_the_weighted_groups(self, workdir, monkeypatch):
        rng = np.random.default_rng(7)
        other = ("non-reactive", "plus")
        a = {**_record(rng, "entry", 1)}
        b = {**_record(rng, "entry", 2), "group": other}
        sa.flush_chunk({GROUP: [a], other: [b]}, 0)
        _set_runs(monkeypatch, workdir, {"entry": ({1: 2.0, 2: 3.0}, 1.0)})
        weights = sa.run_weights()
        merged = sa.merge_groups([GROUP, other], weights)
        assert sorted(merged["path_num"].tolist()) == [1, 2]
        np.testing.assert_allclose(merged["hist2d"],
                                   2.0 * a["hist2d"] + 3.0 * b["hist2d"], rtol=1e-6)


class TestSameResultAsBefore:
    def test_the_profile_equals_the_one_with_weights_baked_in(self, workdir, monkeypatch):
        """Applying w later changes where it happens, not what comes out."""
        rng = np.random.default_rng(8)
        recs = [_record(rng, "entry", pn) for pn in range(1, 7)]
        weights = {pn: float(w) for pn, w in zip(range(1, 7), rng.uniform(0.1, 2, 6))}
        sa.flush_chunk({GROUP: recs}, 0)
        _set_runs(monkeypatch, workdir, {"entry": (weights, 1.0)})
        new = sa.weighted_profile(sa.load_group(GROUP, sa.run_weights()),
                                  rng=np.random.default_rng(1))

        baked = {key: np.vstack([weights[r["path_num"]] * r[key] for r in recs])
                 for key in sa._PER_PATH_SUMS}
        old = sa.weighted_profile(baked, rng=np.random.default_rng(1))

        for key in ("mean_tot", "mean_pol", "exposure", "sd_tot",
                    "ci_lo_tot", "ci_hi_tot"):
            np.testing.assert_allclose(new[key], old[key], rtol=1e-9, equal_nan=True)

    def test_new_weights_change_the_profile_without_touching_the_chunks(self, workdir, monkeypatch):
        rng = np.random.default_rng(9)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 1), _record(rng, "entry", 2)]}, 0)
        (fp,) = sa._group_chunk_files(GROUP)
        before = open(fp, "rb").read()

        _set_runs(monkeypatch, workdir, {"entry": ({1: 1.0, 2: 1.0}, 1.0)})
        a = sa.weighted_profile(sa.load_group(GROUP, sa.run_weights()), n_bootstrap=0)
        _set_runs(monkeypatch, workdir, {"entry": ({1: 1.0, 2: 40.0}, 1.0)})
        b = sa.weighted_profile(sa.load_group(GROUP, sa.run_weights()), n_bootstrap=0)

        assert not np.allclose(a["mean_tot"], b["mean_tot"], equal_nan=True)
        assert open(fp, "rb").read() == before


class TestReusingIntermediates:
    def _params(self, monkeypatch, **overrides):
        monkeypatch.setattr(sa, "RUNS", [{"name": "entry", "mirror_z": False},
                                         {"name": "escape", "mirror_z": True}])
        params = sa.computation_params()
        params.update(overrides)
        return params

    def test_unchanged_settings_reuse_every_current_chunk(self, workdir, monkeypatch):
        rng = np.random.default_rng(10)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 1), _record(rng, "escape", 1)]}, 0)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 2)]}, 4)
        params = self._params(monkeypatch)
        done, next_chunk = sa.prepare_intermediates(params, params)
        assert done == {("entry", 1), ("escape", 1), ("entry", 2)}
        assert next_chunk == 5

    def test_older_format_chunks_are_removed_and_recomputed(self, workdir, monkeypatch):
        rng = np.random.default_rng(11)
        sa.flush_chunk({GROUP: [_record(rng, "entry", 1)]}, 0)
        old = os.path.join(sa.INTERMEDIATE_DIR, f"0001__{sa.group_tag(GROUP)}__sasa.npz")
        np.savez_compressed(old, w=np.ones((1, N_Z)), path_num=np.array([2]))
        params = self._params(monkeypatch)
        done, _ = sa.prepare_intermediates(params, params)
        assert done == {("entry", 1)}
        assert not os.path.exists(old)

    @pytest.mark.parametrize("change", [{"probe_radius": 1.2}, {"n_sphere_points": 480},
                                        {"z_bin_width": 0.5}])
    def test_a_changed_setting_recomputes_everything(self, workdir, monkeypatch, change):
        sa.flush_chunk({GROUP: [_record(np.random.default_rng(12), "entry", 1)]}, 0)
        previous = self._params(monkeypatch)
        current = self._params(monkeypatch, **change)
        assert sa.prepare_intermediates(current, previous) == (set(), 0)
        assert sa._group_chunk_files(GROUP) == []

    def test_recompute_discards_everything(self, workdir, monkeypatch):
        sa.flush_chunk({GROUP: [_record(np.random.default_rng(13), "entry", 1)]}, 0)
        params = self._params(monkeypatch)
        assert sa.prepare_intermediates(params, params, recompute=True) == (set(), 0)

    def test_no_record_of_the_settings_means_no_reuse(self, workdir, monkeypatch):
        sa.flush_chunk({GROUP: [_record(np.random.default_rng(14), "entry", 1)]}, 0)
        assert sa.prepare_intermediates(self._params(monkeypatch), None) == (set(), 0)

    def test_mirror_only_matters_for_runs_in_both(self, monkeypatch):
        previous = self._params(monkeypatch)
        added = {**previous, "mirror_z": {**previous["mirror_z"], "internal": True}}
        flipped = {**previous, "mirror_z": {"entry": True, "escape": True}}
        assert sa.changed_settings(previous, added) == []
        assert sa.changed_settings(previous, flipped) == ["mirror_z[entry]"]

    def test_the_settings_survive_the_meta_file(self, workdir, monkeypatch):
        import json
        params = self._params(monkeypatch)
        with open(workdir / sa.META_FILE, "w") as fh:
            json.dump({"computation": params, "z_range": [0, 1]}, fh)
        assert sa.changed_settings(sa.read_previous_params(), params) == []

    def test_nothing_is_recomputed_when_every_path_is_done(self, monkeypatch, capsys):
        jobs = [(1, "l", "m", "t", "entry", False, GROUP), (2, "l", "m", "t", "entry", False, GROUP)]
        monkeypatch.setattr(sa, "build_job_list", lambda bin_info: (jobs, []))
        sa.parse_all({}, done={("entry", 1), ("entry", 2)})
        out = capsys.readouterr().out
        assert "2 of 2 paths are already in the intermediates" in out
        assert "Nothing new to compute" in out

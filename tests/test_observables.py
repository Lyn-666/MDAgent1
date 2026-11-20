from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdagent.agent import LoopSettings, decide
from mdagent.observables import DistanceSpec, diagnose_distances, distance_atoms, parse_distances, two_state

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"


def test_distance_specs_are_read_and_checked():
    s = DistanceSpec.parse("A:88:OH-A:111:CG1@0.6")
    assert s.a == ("A", 88, "OH") and s.b == ("A", 111, "CG1") and s.cutoff_nm == 0.6
    assert s.label == "A:88:OH-A:111:CG1@0.6"
    assert DistanceSpec.parse("88:OH - 111:CG1 @ 0.6").label == "88:OH-111:CG1@0.6"   # Amber: no chain.
    assert [x.label for x in parse_distances("A:1:CA-A:2:CA@1;A:3:CB-A:4:CB@0.8")] == \
        ["A:1:CA-A:2:CA@1", "A:3:CB-A:4:CB@0.8"]
    for bad in ["A:88:OH A:111:CG1@0.6", "A:88:OH-A:111:CG1", "A:88:OH-A:111:CG1@60"]:
        with pytest.raises(ValueError):
            DistanceSpec.parse(bad)
    top = md.load_topology(str(REPO / "examples" / "181L.pdb"))
    assert len(distance_atoms(top, [s])) == 1
    with pytest.raises(ValueError, match="not in the structure"):
        distance_atoms(top, [DistanceSpec.parse("A:88:XX-A:111:CG1@0.6")])


def test_two_states_need_to_cross_the_buffer():
    v = np.array([0.40, 0.58, 0.62, 0.70, 0.62, 0.50, 0.70])
    assert list(two_state(v, 0.6, 0.05)) == [0, 0, 0, 1, 1, 0, 1]   # 0.58 and 0.62 keep the last state.


def _traj(values_nm):
    """Two atoms on the x axis at the given distances."""
    top = md.Topology()
    chain = top.add_chain()
    for name in ("A", "B"):
        res = top.add_residue("UNK", chain)
        top.add_atom(name, md.element.carbon, res)
    xyz = np.zeros((len(values_nm), 2, 3))
    xyz[:, 1, 0] = values_nm
    return md.Trajectory(xyz, top)


def test_distances_are_judged_like_side_chains():
    rng = np.random.default_rng(0)
    switching = [_traj(np.where(np.sin(np.arange(4000) / 40 + r) > 0, 0.4, 0.9) + rng.normal(0, 0.02, 4000))
                 for r in range(3)]
    stuck = [_traj(0.4 + rng.normal(0, 0.02, 4000)) for _ in range(3)]
    spec = [DistanceSpec.parse("A:1:A-A:2:B@0.6")]
    ok = diagnose_distances(switching, spec, [(0, 1)], 10.0, precision=0.1)[0]
    assert ok.verdict == "ok" and ok.quantity == "distance" and abs(ok.populations["near"] - 0.5) < 0.1
    assert ok.label == "A:1:A-A:2:B@0.6" and ok.extra["cutoff_nm"] == 0.6
    trapped = diagnose_distances(stuck, spec, [(0, 1)], 10.0, precision=0.1)[0]
    # Trapped: not done, and the error is not 0 any more (error floor): nothing was seen to change.
    assert trapped.verdict == "no_flip" and trapped.standard_error["near"] > 0.1

    # No enhanced sampling for distances: a stuck distance is "stalled", not "enhanced_sampling".
    loop = LoopSettings(max_ns=20.0, no_flip_patience_ns=10.0)
    d, next_ns = decide([trapped], 11.0, loop, may_switch=True, bias_available=False)
    assert d[0].action == "extend" and d[0].stalled and next_ns > 0
    d, next_ns = decide([trapped], 20.0, loop, may_switch=True, bias_available=False)
    assert d[0].action == "stalled" and d[0].cause == "stalled" and next_ns == 0


def test_command_line_checks_the_distance_atoms_before_building(tmp_path, capsys):
    from mdagent.cli import main

    base = ["run", "--pdb", str(REPO / "examples" / "181L.pdb"), "--out", str(tmp_path / "r"), "--accept-problems"]
    assert main(base) == 2 and "--residues or --around-ligand" in capsys.readouterr().err
    assert main(base + ["--distances", "A:88:XX-A:111:CG1@0.6"]) == 2
    assert "A:88:XX is not in the structure" in capsys.readouterr().err


@pytest.mark.slow
def test_plain_md_on_a_distance(tmp_path, capsys):
    """Tiny peptide on CPU, quantity distance: plain MD runs, no enhanced sampling, a stop reason is kept."""
    from dataclasses import asdict

    from mdagent.agent import Goal, RunState, run_all
    from mdagent.build import SetupOptions, build_system
    from mdagent.messages import read
    from mdagent.simulate import MDSettings, SystemFiles, write_inputs

    built = build_system(DATA / "fragment_108-114.pdb", SetupOptions(forcefield="amber14", padding_nm=0.8))
    write_inputs(built, tmp_path)
    state = RunState(pdb="x", target_description="distance", target=[("A", 108), ("A", 114)], n_replicas=2,
                     base_seed=1, md=MDSettings(equilibration_ps=1.0, save_interval_ps=0.5, platform="CPU"),
                     loop=LoopSettings(first_ns=0.002, extend_ns=0.002, max_ns=0.004, no_flip_patience_ns=0.003),
                     goal=asdict(Goal(quantity="distance_states")), distances=["A:108:CA-A:114:CA@1.0"])
    state.save(tmp_path)
    decisions, labels = run_all(tmp_path, state, SystemFiles(built.topology, built.positions, built.system),
                                log=lambda line: None)
    assert labels == [] and [d.residue for d in decisions] == ["A:108:CA-A:114:CA@1"]
    saved = RunState.load(tmp_path)
    assert saved.notes["plain_md_outcome"]["reason"] in ("target_reached", "stalled", "budget_exhausted")
    assert not (tmp_path / "enhanced").exists()
    import json
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["residues"][0]["name"] == "A:108:CA-A:114:CA@1" and status["residues"][0]["populations"]
    assert read(tmp_path)  # The chat was told how plain MD ended.
    # `mdagent diagnose` judges the distance, like the run did, not the chi1 of the two residues.
    from mdagent.cli import main

    capsys.readouterr()
    assert main(["diagnose", "--run", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert "A:108:CA-A:114:CA@1" in shown and "108" not in shown.replace("A:108:CA-A:114:CA@1", "")

import json

from mdagent.report import compare, final_results, write


def _run(tmp_path, name, plain_pop=0.70, plain_se=0.02, enh=True, verdict="reached", enh_se=1.0):
    run = tmp_path / name
    (run / "enhanced").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "pdb": "examples/181L.pdb", "target_description": "residues within 5 Å of BNZ", "n_replicas": 3,
        "rounds_ns": [1.0, 5.0], "md": {"temperature_k": 300.0, "equilibration_ps": 600.0},
        "loop": {"first_ns": 1.0, "extend_ns": 5.0, "max_ns": 20.0},
        "setup": {"forcefield": "charmm36m", "ph": 7.0, "ion_conc_molar": 0.15, "cation": "Na+",
                  "box_shape": "cube", "padding_nm": 1.0},
        "goal": {"quantity": "chi1_populations", "description": "Val111 rotamers", "precision": 0.05,
                 "permissions": ["switch_method", "stop_early"]},
        "notes": {"plain_md_outcome": {"reason": "stalled", "ns_per_replica": 6.0, "budget_ns": 20.0},
                  "actions": [{"time": "t", "phase": "plain_md", "action": "add_replicas", "executed": False,
                               "reason": "spread"}]},
        "enhanced": {"settings": {"n_replicas": 2},
                     "outcome": {"reason": "target_reached", "ns_per_replica": 5.0, "budget_ns": 20.0}}}))
    (run / "status.json").write_text(json.dumps({"residues": [
        {"chain": "A", "resseq": 88, "resname": "TYR", "name": "", "verdict": "ok", "transitions": [9, 8, 7],
         "populations": {"g+": 0.1, "t": plain_pop, "g-": 0.2}, "standard_error": {"g+": 0.01, "t": plain_se, "g-": 0.02}},
        {"chain": "A", "resseq": 111, "resname": "VAL", "name": "", "verdict": "no_flip", "transitions": [0, 0, 0],
         "populations": {"g+": 0.0, "t": 1.0, "g-": 0.0}, "standard_error": {"g+": 0.0, "t": 0.0, "g-": 0.0}}]}))
    if enh:
        (run / "enhanced" / "populations.json").write_text(json.dumps({"n_replicas": 2, "ns_per_replica": 5.0,
            "residues": [{"residue": "VAL111", "populations": {"g+": 0.3, "t": 0.6, "g-": 0.1},
                          "standard_error": {"g+": 0.03 * enh_se, "t": 0.04 * enh_se, "g-": 0.02 * enh_se},
                          "verdict": verdict, "transitions_per_replica": [180, 175]}]}))
    return run


def test_final_results_prefer_the_reweighted_enhanced_result(tmp_path):
    res = {r["label"]: r for r in final_results(_run(tmp_path, "a"))}
    assert res["TYR88"]["source"] == "plain_md" and res["TYR88"]["met"]
    assert res["VAL111"]["source"] == "enhanced_sampling" and res["VAL111"]["populations"]["t"] == 0.6


def test_report_says_what_was_asked_found_and_left_open(tmp_path):
    zh, en = write(_run(tmp_path, "a", verdict="not_yet"))
    text_zh, text_en = zh.read_text(), en.read_text()
    assert "Val111 rotamers" in text_en and "standard error of every population at most 0.05" in text_en
    assert "| VAL111 | g+ 0.30 ± 0.03, t 0.60 ± 0.04, g- 0.10 ± 0.02 | enhanced sampling (reweighted) | no |" in text_en
    assert "1 items reached the requested precision and 1 did not: VAL111" in text_en
    assert "- Plain MD: sampling stalled, after 6 ns per replica (budget 20 ns)" in text_en
    assert "add_replicas (suggested)" in text_en and "Ligands were removed" in text_en
    assert "- VAL111: not precise yet" in text_en
    assert "it means trapped" not in text_en              # VAL111's final result came from enhanced sampling.
    assert "report.zh.md, report.en.md" in text_en
    # The report says what the run cost, so that a cheaper run can be compared with it.
    assert "Cost: 28 ns of production" in text_en and "1.8 ns to equilibrate 3 replicas" in text_en
    assert "标准误差不超过 0.05" in text_zh
    assert "普通 MD：采样停滞" in text_zh and "配体被去掉了" in text_zh


def test_compare_asks_whether_two_runs_give_the_same_answer(tmp_path):
    """The tolerance is what counts, not whether a difference is significant.

    An imprecise run must not pass: with |z| <= 2 a run with large errors agreed with
    everything, which is exactly what a short run has to prove it does not do.
    """
    a = _run(tmp_path, "a", plain_se=0.005, enh_se=0.1)
    near = _run(tmp_path, "b", plain_pop=0.72, plain_se=0.005, enh_se=0.1)   # 0.02 apart, errors small.
    far = _run(tmp_path, "c", plain_pop=0.85, plain_se=0.005, enh_se=0.1)    # 0.15 apart: a real difference.
    vague = _run(tmp_path, "d", plain_pop=0.70, plain_se=0.05)               # Same value, far too imprecise.
    assert compare(a, near)[1]
    assert not compare(a, far)[1]
    assert not compare(a, vague)[1]
    assert "TYR88" in compare(a, far)[0]
    # A larger tolerance makes the 0.15 difference acceptable again.
    assert compare(a, far, tolerance=0.3)[1]

"""Step 6: the result report a run delivers, and a comparison of two runs (for step 5).

At the end of a run, report.zh.md and report.en.md are written in the run folder. They say
what was asked, what was simulated, the result of every item with its error and whether it
meets the goal, why each phase stopped, which actions the agent took, how far the result can
be trusted, where the files are, and what is still open. Everything comes from the files of
the run (run.json, status.json, enhanced/populations.json); nothing is made up.

    python -m mdagent report --run runs/x                 # write the report again
    python -m mdagent compare --run runs/x --reference runs/long

compare asks whether two runs give the same answer, with an equivalence test: the two
results are equivalent when the whole confidence interval of their difference lies inside
the tolerance the user asked for (|difference| + z * combined error <= tolerance). This is
the question that matters for a cheaper run: "is the answer still right?". It is not the
same as the usual significance test. |z| <= 2 only says that a difference was not proven;
with few data every difference passes that test, so a short run would look "right" exactly
because it is imprecise. z is still printed, because it says which way a difference goes.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import NormalDist

from . import __version__
from .outputs import describe, host

REASONS = {
    "target_reached": ("达到目标", "goal reached"),
    "stalled": ("采样停滞", "sampling stalled"),
    "not_reachable": ("上限内达不到", "not reachable within the limit"),
    "budget_exhausted": ("预算用完", "budget used up"),
    "insufficient_evidence": ("证据不足", "insufficient evidence"),
    "failed": ("运行失败", "run failed"),
}


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def final_results(run_dir: Path) -> list[dict]:
    """The final number of every item: the reweighted enhanced sampling result where there is one,
    otherwise plain MD. Each: label, source, populations, standard_error, met (goal met), verdict, transitions."""
    run_dir = Path(run_dir)
    out = {}
    for r in _load(run_dir / "status.json").get("residues", []):
        if not r.get("populations"):
            continue
        label = r.get("name") or f"{r['resname']}{r['resseq']}"
        out[label] = {"label": label, "source": "plain_md", "populations": r["populations"],
                      "standard_error": r["standard_error"], "verdict": r["verdict"], "met": r["verdict"] == "ok",
                      "transitions": r["transitions"], "extra": r.get("extra")}
    for r in _load(run_dir / "enhanced" / "populations.json").get("residues", []):
        out[r["residue"]] = {"label": r["residue"], "source": "enhanced_sampling", "populations": r["populations"],
                             "standard_error": r["standard_error"], "verdict": r.get("verdict", ""),
                             "met": r.get("verdict") == "reached", "transitions": r["transitions_per_replica"],
                             "free_energy_kcal": r.get("free_energy_kcal")}
    return list(out.values())


def _cost(state: dict):
    from .cost import of_run

    return of_run(state)


def _backbone_items(state: dict) -> list:
    """Backbone item specs of a run.json (runs from before the merge kept a settings dict there)."""
    b = state.get("backbone")
    if isinstance(b, list):
        return b
    return [f"rama:{c}:{n}" if c else f"rama:{n}" for c, n in state.get("target", [])]


def _precision(state: dict):
    """The precision the run asked for, read back from run.json."""
    from .uncertainty import Precision, planned_looks

    goal, loop = state.get("goal") or {}, state.get("loop") or {}
    if goal.get("allocation") == "projection":   # as RunState.precision(): calibrated for the stop looks
        return Precision(epsilon=goal.get("precision", 0.05), criterion="confidence", alpha=goal.get("alpha", 0.05),
                         looks=goal.get("stop_looks", 3))
    factor = goal.get("extend_factor", 2.0) if "extend" in (goal.get("permissions") or []) else 1.0
    return Precision(epsilon=goal.get("precision", 0.05), criterion=goal.get("criterion", "standard_error"),
                     alpha=goal.get("alpha", 0.05),
                     looks=planned_looks(loop.get("first_ns", 1.0), loop.get("extend_ns", 5.0),
                                         loop.get("max_ns", 20.0), factor))


def _status(run_dir: Path) -> dict:
    return {(r.get("name") or f"{r['resname']}{r['resseq']}"): r
            for r in _load(Path(run_dir) / "status.json").get("residues", [])}


def _backbone_section(run_dir: Path, state: dict, T) -> list:
    """Backbone basins: each observable with its interval and whether it converged, and the landscapes."""
    if (state.get("goal") or {}).get("quantity") != "backbone_basins":
        return []
    from .basins import judge

    p = _precision(state)
    z, tol = p.z, p.epsilon
    status = _status(run_dir)
    L = [T("## 主链构象区：每个量单独判断", "## Backbone basins: every observable judged on its own"), "",
         T(f"区间是 {1 - p.alpha:.0%} 置信区间的半宽（区间系数 {z:.2f}）。“已收敛”是 agent 只凭自己的轨迹作出的判断；"
           "一个量没收敛，就单独写明，不和其他量一起宣布收敛。",
           f"Intervals are half-widths of the {1 - p.alpha:.0%} interval (factor {z:.2f}). \"Converged\" is the "
           "agent's judgement from its own trajectories only; an observable that did not converge is said so on "
           "its own, never claimed together with the others."), ""]
    for label in _backbone_items(state):
        r = status.get(label)
        if not r or not r.get("populations"):
            continue
        x = r.get("extra") or {}
        est = {"transitions": r["transitions"], "se": r["standard_error"], "rates": x.get("rates", {}),
               "dg_se": x.get("dg_se") if x.get("dg_se") is not None else float("inf"),
               "split_js": x.get("split_js", 1.0)}
        ok = judge(est, z, tol, min_transitions=(state.get("loop") or {}).get("min_flips_per_replica", 4))
        yes, no = T("已收敛", "converged"), T("未收敛", "not converged")
        pops = ", ".join(f"{k} {v:.3f} ± {z * r['standard_error'][k]:.3f}" for k, v in r["populations"].items())
        names = list(r["populations"])
        dg = x.get("dg_kt")
        finite = dg is not None and x.get("dg_se") is not None and math.isfinite(dg) and math.isfinite(x["dg_se"])
        dg_txt = (f"{dg:.2f} ± {z * x['dg_se']:.2f} kT" if finite
                  else T("无法估计（有构象区没被访问）", "cannot be estimated (a basin was never visited)"))
        L += [T(f"### {label}（{'，'.join(x.get('angles', []))}）", f"### {label} ({', '.join(x.get('angles', []))})"), "",
              T("| 量 | 结果 | 判断 |", "| Observable | Result | Judgement |"), "|---|---|---|",
              f"| {T('各构象区比例', 'Basin populations')} | {pops} | {yes if ok['populations'] else no} |",
              f"| {T('自由能差', 'Free-energy difference')} G({names[1]}) − G({names[0]}) | {dg_txt} | "
              f"{yes if ok['dg'] else no} |"]
        for k, rate in (x.get("rates") or {}).items():
            rel = rate.get("rel_se")
            err = f" ± {z * rel * rate['k_per_ns']:.3g}" if rel is not None and rel != float("inf") else ""
            L.append(f"| {T('速率', 'Rate')} {k} | {rate['k_per_ns']:.3g}{err} /ns "
                     f"({T('观察到', 'seen')} {rate['n']} {T('次', 'times')}) | {yes if ok['rates'] else no} |")
        L.append(f"| {T('φ/ψ 自由能图', 'phi/psi free-energy landscape')} | "
                 f"![{label}](landscape_{label.replace(':', '_')}.svg) | {yes if ok['landscape'] else no} "
                 f"({T('两半数据的 JS 散度', 'split-half JS divergence')} {x.get('split_js', float('nan')):.3f} bit) |")
        L += ["", T(f"每条重复的状态变化次数：{r['transitions']}。少于 4 次的重复不足以支撑比例的估计。",
                    f"State changes per replica: {r['transitions']}. A replica with fewer than 4 cannot support "
                    "a population estimate."), ""]
    return L


def _selection_zh(text: str) -> str:
    """Chinese wording of stability.atom_selection's description."""
    import re

    m = re.match(r"(C-alpha|heavy) atoms of (.*) \((\d+) atoms(?:; fewer than (\d+) residues)?\)$", text)
    if not m:
        return text
    kind, region, n, few = m.groups()
    region = "整个蛋白" if region == "the whole protein" else region
    return (f"{region}的{'Cα 原子' if kind == 'C-alpha' else '重原子'}（{n} 个原子"
            + (f"；残基少于 {few} 个，所以用重原子" if few else "") + "）")


def _stability_section(run_dir: Path, state: dict, T) -> list:
    """Structure stability: means, steady state, native-like time, the departure bound, figures."""
    if (state.get("goal") or {}).get("quantity") != "structure_stability":
        return []
    s = _load(run_dir / "stability" / "summary.json")
    if not s:
        return []
    items, nat, st = s.get("items", {}), s.get("native", {}), s.get("settings", {})
    L = [T("## 结构稳定性", "## Structure stability"), "",
         T(f"原子：{_selection_zh(s.get('selection', ''))}。参照：起始结构（`protein.pdb`），每帧先叠合再算 RMSD。"
           + ("每条轨迹开头 " if s.get('burn_in_ns') else "")
           + (f"{s.get('burn_in_ns', 0):g} ns 被判为还没有平衡，没有算进结果（自动平衡检测）。" if s.get('burn_in_ns') else
              "自动平衡检测没有发现需要丢掉的开头，所有帧都算进了结果。"),
           f"Atoms: {s.get('selection', '')}. Reference: the starting structure (`protein.pdb`); every frame is "
           "superposed before the RMSD. "
           + (f"The first {s.get('burn_in_ns', 0):g} ns of each replica were judged not yet equilibrated and left "
              "out (automated equilibration detection)." if s.get('burn_in_ns') else
              "Automated equilibration detection found no start to leave out, so every frame counts.")), "",
         T("| 量 | 平均值 ± 区间半宽 | 各条轨迹的平均值 | 后半段 − 前半段 | 稳态 | 判断 |",
           "| Quantity | Mean ± interval half-width | Mean per replica | Second half − first half | Steady | Verdict |"),
         "|---|---|---|---|---|---|"]
    for name, x in items.items():
        dr = x.get("drift") or {}
        per = ", ".join(f"{v:.3f}" for v in x.get("per_replica", []) if v is not None)
        d = dr.get("drift_nm")
        L.append(f"| {name} | {x.get('mean', float('nan')):.3f} ± {(x.get('halfwidth_nm') or float('nan')):.3f} nm | {per} | "
                 + (f"{d:+.3f} ± {(dr.get('halfwidth_nm') or float('nan')):.3f} nm" if d is not None else "-")
                 + f" | {T('是', 'yes') if dr.get('steady') else T('否', 'no')} | {x.get('verdict', '')} |")
    cut = st.get("rmsd_cutoff_nm", 0.3)
    if nat:
        frac = nat.get("fraction")
        L += ["", T(f"**是否保持起始结构：** RMSD 低于 {cut:g} nm 的时间占 {frac:.0%}（各条轨迹："
                    f"{', '.join(f'{f:.0%}' for f in nat.get('fraction_per_replica', []) if f is not None)}）；"
                    f"离开的次数：{nat.get('departures_per_replica')}。",
                    f"**Does it keep its starting structure?** RMSD below {cut:g} nm for {frac:.0%} of the time "
                    f"(per replica: {', '.join(f'{f:.0%}' for f in nat.get('fraction_per_replica', []) if f is not None)}); "
                    f"departures: {nat.get('departures_per_replica')}.")]
        if nat.get("departure_rate_upper_per_us") is not None:
            k = sum(nat.get("departures_per_replica") or [])
            goal_rate = nat.get("max_departure_rate_per_us", 10.0)
            upper = nat["departure_rate_upper_per_us"]
            met = upper <= goal_rate
            seen_zh = "没有一次离开" if k == 0 else f"离开了 {k} 次"
            seen_en = "No departure" if k == 0 else f"{k} departure(s)"
            L.append(T(f"所有轨迹合计 {nat.get('watched_ns', 0):.3g} ns 里{seen_zh}。按 95% 置信上限，离开的速率不超过每 µs "
                       f"{upper:.3g} 次；目标是不超过每 µs {goal_rate:g} 次，"
                       + ("已经达到。" if met else "还没有达到，要跑得更久。")
                       + "这个上限说的是离开有多快，不能证明结构永远不离开。",
                       f"{seen_en} in {nat.get('watched_ns', 0):.3g} ns of all replicas together. At 95 % confidence "
                       f"the departure rate is at most {upper:.3g} per µs; the goal is at most {goal_rate:g} per µs, "
                       + ("which is met. " if met else "which is not met yet: it needs a longer run. ")
                       + "The bound says how fast a departure could be; it does not prove the structure never leaves."))
    L += ["", T("RMSD 和 Rg 随时间的变化（阴影部分没有算进结果）：", "RMSD and Rg against time (the shaded start was left out):"),
          "", "![RMSD and Rg](stability/rmsd_rg.svg)", ""]
    if (run_dir / "stability" / "fes_rmsd_rg.svg").exists():
        L += [T("RMSD–Rg 自由能面（颜色越深自由能越低；每格的数值和误差在 `stability/fes_rmsd_rg.csv`）：",
                "RMSD-Rg free-energy surface (darker is lower; values and errors per bin in `stability/fes_rmsd_rg.csv`):"),
              "", "![F(RMSD, Rg)](stability/fes_rmsd_rg.svg)", ""]
    return L


def _cluster_section(run_dir: Path, T) -> list:
    c = _load(run_dir / "clusters" / "clusters.json")
    if not c:
        return []
    L = [T("## 构象聚类", "## Conformational clusters"), "",
         T(f"方法：GROMOS（Daura 1999），叠合后的 RMSD 阈值 {c['cutoff_nm']:g} nm，用了平衡后的 {c['frames_used']} 帧"
           + ("（所有轨迹合在一起，用了每一帧）。" if c['stride'] == 1 else f"（所有轨迹合在一起，每 {c['stride']} 帧取 1 帧）。")
           + "误差来自各条轨迹之间的差别。代表结构可以用 PyMOL 或 VMD 打开。",
           f"Method: GROMOS (Daura 1999), RMSD cutoff {c['cutoff_nm']:g} nm after superposition, {c['frames_used']} "
           + ("equilibrated frames (all replicas, every frame). " if c['stride'] == 1 else
              f"equilibrated frames (all replicas, every {c['stride']}th frame). ") + "Errors come from the spread between "
           "replicas. Open the representative structures in PyMOL or VMD."), "",
         T("| 簇 | 占比 ± 误差 | 出现在哪些轨迹 | 代表结构 |", "| Cluster | Population ± error | Replicas | Representative |"),
         "|---|---|---|---|"]
    for k in c.get("clusters", []):
        err = k.get("error")
        L.append(f"| {k['cluster']} | {k['population']:.0%}" + (f" ± {err:.0%}" if err is not None else "")
                 + f" | {', '.join(map(str, k['replicas_visiting']))} | `{k.get('pdb', '')}` |")
    if c.get("rest_population"):
        L.append(T(f"\n其余 {c['n_clusters'] - len(c['clusters'])} 个小簇合计占 {c['rest_population']:.0%}。",
                   f"\nThe other {c['n_clusters'] - len(c['clusters'])} small clusters hold {c['rest_population']:.0%} together."))
    L.append("")
    return L


def _decision_section(state: dict, T) -> list:
    """Projection allocation: why it started as it did, what it saw, how it allocated, why it stopped, the cost."""
    goal, notes = state.get("goal") or {}, state.get("notes") or {}
    if goal.get("allocation") != "projection":
        return []
    loop = state.get("loop") or {}
    start = notes.get("replicas_at_start", state.get("n_replicas"))
    budget = goal.get("budget_ns") or start * loop.get("max_ns", 0)
    L = [T("## agent 的决策", "## The agent's decisions"), "",
         T(f"- **开始：** {start} 条重复，每条 {loop.get('first_ns', 0):g} ns。这一次查看只用来规划，不允许停，"
           f"所以不消耗显著性水平。之后最多再看 {goal.get('stop_looks', 3)} 次，区间按这个次数校准。"
           f"限制：总预算 {budget:g} ns，每条最多 {loop.get('max_ns', 0):g} ns，最多 {goal.get('max_replicas', 10)} 条。",
           f"- **Start:** {start} replicas of {loop.get('first_ns', 0):g} ns. This look only plans and may not "
           f"stop, so it spends none of alpha. After it, at most {goal.get('stop_looks', 3)} looks, with intervals "
           f"calibrated for that many. Limits: {budget:g} ns in total, {loop.get('max_ns', 0):g} ns per replica, "
           f"{goal.get('max_replicas', 10)} replicas.")]
    for a in notes.get("actions", []):
        if a.get("phase") == "plain_md":
            L.append(f"- **{T('查看', 'Look')}** ({a.get('time', '')}): {a['reason']}")
    d = notes.get("allocation_decision")
    if d:
        L.append(T(f"- **停止：** {d['decision']}。{d['reason']}。",
                   f"- **Stop:** {d['decision']}. {d['reason'][0].upper() + d['reason'][1:]}."))
    c = _cost(state)
    used = c.aggregate_ns
    L += ["", T(f"**成本：** 生产 {used:g} ns（所有重复相加），加上平衡共 {c.compute_ns:g} ns"
                + (f"，实际用了 {c.measured_seconds / 3600:.1f} 小时" if c.measured_seconds else "")
                + f"。用了预算的 {_pct(used / budget)}" + (f"，比把预算跑满省了 {_pct(1 - used / budget)}" if budget else "")
                + "。这个节省是相对预算的，不是相对“达到同样准确度的固定方案”；后者要用实验 A 那样的基准测试才量得出。",
                f"**Cost:** {used:g} ns of production (all replicas together), {c.compute_ns:g} ns with equilibration"
                + (f", {c.measured_seconds / 3600:.1f} hours of wall-clock" if c.measured_seconds else "")
                + f". That is {_pct(used / budget)} of the budget" + (f", {_pct(1 - used / budget)} less than running it out"
                                                                    if budget else "")
                + ". The saving is against the budget, not against a fixed plan that reaches the same accuracy; "
                  "that needs a benchmark like experiment A.") if budget else "", ""]
    return L


def _pct(x: float) -> str:
    """A share as a percentage that never rounds a small part to 0 % or a large part to 100 %."""
    if 0 < x < 0.01:
        return "<1%"
    if 0.99 < x < 1:
        return ">99%"
    return f"{x:.0%}"


def _fmt(pops: dict, errs: dict) -> str:
    if list(pops) == ["mean"]:      # a continuous mean in nm (structure stability)
        return f"{pops['mean']:.3f} ± {errs['mean']:.3f} nm"
    return ", ".join(f"{k} {pops[k]:.2f} ± {errs[k]:.2f}" for k in pops)


def write(run_dir: Path) -> list[Path]:
    """Write report.zh.md and report.en.md. Returns their paths."""
    run_dir = Path(run_dir)
    state = _load(run_dir / "run.json")
    goal = state.get("goal") or {}
    precision = goal.get("precision", 0.05)
    results = final_results(run_dir)
    met = [r for r in results if r["met"]]
    open_ = [r for r in results if not r["met"]]
    notes = state.get("notes", {})
    outcomes = [("plain_md", notes.get("plain_md_outcome")), ("enhanced_sampling", (state.get("enhanced") or {}).get("outcome")),
                ("any", notes.get("outcome"))]
    setup = state.get("setup") or {}
    if state.get("backbone"):
        try:
            from .landscape import write as write_landscapes

            write_landscapes(run_dir)
        except Exception as e:   # noqa: BLE001 - a missing figure must never stop the report itself.
            print(f"Note: the landscapes could not be drawn ({type(e).__name__}: {e}).")
    if state.get("rounds_ns") and (goal.get("quantity") == "structure_stability" or goal.get("cluster")):
        try:
            from .clustering import run as run_clustering

            s = _load(run_dir / "stability" / "summary.json")
            whole = "the whole protein" in s.get("selection", "")
            run_clustering(run_dir, [] if whole else [tuple(k) for k in state.get("target", [])],
                           len(state.get("rounds_ns", [])),
                           cutoff_nm=goal.get("cluster_cutoff_nm", 0.15), start_ns=s.get("burn_in_ns", 0.0),
                           dt_ns=(state.get("md") or {}).get("save_interval_ps", 5.0) / 1000.0)
        except Exception as e:   # noqa: BLE001 - clustering must never stop the report itself.
            print(f"Note: the conformations could not be clustered ({type(e).__name__}: {e}).")
    files = describe(run_dir)
    paths = []
    for lang in ("zh", "en"):
        zh = lang == "zh"

        def T(a, b):
            return a if zh else b
        L = [T(f"# MDAgent 结果报告：{run_dir.name}", f"# MDAgent result report: {run_dir.name}"), "",
             T(f"写于 {time.strftime('%Y-%m-%d %H:%M')}，电脑“{host()}”，文件夹 {run_dir.resolve()}，MDAgent {__version__}。",
               f"Written {time.strftime('%Y-%m-%d %H:%M')} on computer '{host()}', folder {run_dir.resolve()}, "
               f"MDAgent {__version__}."), ""]

        L += [T("## 问题", "## The question"), ""]
        quantity = goal.get("quantity", "chi1_populations")
        events = quantity == "event_seen"
        what = (T(f"能否看到这些变化：{', '.join(state.get('events', []))}",
                  f"whether these changes are seen: {', '.join(state.get('events', []))}") if events else
                T("原子间距离处在“近”和“远”两种状态的比例", "populations of the near and far states of atom distances")
                if quantity == "distance_states" else
                T(f"主链构象区（{', '.join(_backbone_items(state))}）：各构象区比例、构象区之间的转变统计（速率和停留时间）、"
                  "自由能差和 φ/ψ 自由能面",
                  f"backbone basins ({', '.join(_backbone_items(state))}): basin populations, transition statistics "
                  "between basins (rates and dwell times), free-energy difference and the free-energy landscape")
                if quantity == "backbone_basins" else
                T("结构稳定性：相对起始结构的骨架 RMSD 和回转半径（Rg）是否达到稳态、平均值和误差、保持天然结构的时间比例、"
                  "RMSD–Rg 自由能面和构象聚类",
                  "structure stability: whether the backbone RMSD from the start and the radius of gyration (Rg) reach a "
                  "steady state, their means and errors, the time spent native-like, the RMSD-Rg free-energy surface "
                  "and conformational clusters")
                if quantity == "structure_stability" else
                T("目标区域侧链的 chi1 构象（g+、t、g-）的比例", "populations of the chi1 states (g+, t, g-) of the target side chains"))
        L.append(T(f"- 目标量：{what}。", f"- Quantity: {what}."))
        if goal.get("description"):
            L.append(T(f"- 用户的原话：{goal['description']}", f"- In the user's words: {goal['description']}"))
        if events:
            L.append(T(f"- 怎样算看到：从别的状态进入目标状态，并停留至少 {goal.get('hold_ps', 100):g} ps；"
                       f"在 {goal.get('seen_in', 1)} 条重复里看到就停。",
                       f"- What counts as seen: the item enters the target state from another state and stays at "
                       f"least {goal.get('hold_ps', 100):g} ps; it stops once seen in {goal.get('seen_in', 1)} "
                       "replica(s)."))
        elif quantity == "structure_stability":
            tol = goal.get("stability_tolerance_nm", 0.02)
            L.append(T(f"- 稳定的标准：离开起始结构的速率（95% 上限）不超过每 µs {goal.get('max_departure_rate_per_us', 10.0):g} 次。",
                       f"- What counts as stable: the rate of leaving the start structure is at most "
                       f"{goal.get('max_departure_rate_per_us', 10.0):g} per µs (95 % upper bound)."))
            L.append(T(f"- 要求的精度：RMSD 和 Rg 平均值的区间半宽不超过 {tol:g} nm；稳态：后半段和前半段平均值之差加上它的区间半宽，"
                       f"不超过 {3 * tol:g} nm。RMSD 超过 {goal.get('rmsd_cutoff_nm', 0.3):g} nm 算离开起始结构。",
                       f"- Requested precision: interval half-width of the mean RMSD and Rg at most {tol:g} nm; steady "
                       f"state: the second-half mean minus the first-half mean, plus its half-width, at most {3 * tol:g} "
                       f"nm. An RMSD above {goal.get('rmsd_cutoff_nm', 0.3):g} nm counts as departed from the start."))
        else:
            L.append(T(f"- 要求的精度：{_precision(state).describe(zh=True)}。",
                       f"- Requested precision: {_precision(state).describe()}."))
        if quantity == "backbone_basins":
            L.append(T(f"- 转变速率：主要转变的相对误差不超过 {goal.get('rate_rel_error', 0.25):g}；自由能面：低自由能区域里"
                       f"每格误差不超过 {goal.get('fes_error_kcal', 0.5):g} kcal/mol。",
                       f"- Transition rates: relative error of the leading transitions at most "
                       f"{goal.get('rate_rel_error', 0.25):g}; landscape: error of every low free-energy bin at most "
                       f"{goal.get('fes_error_kcal', 0.5):g} kcal/mol."))
            from .backbone import BackboneSettings, DEFAULT_BASINS_TEXT
            bb = BackboneSettings.from_dict(state.get("backbone_settings")
                                            or (state.get("backbone") if isinstance(state.get("backbone"), dict) else {}))
            if bb.rama == "centres":
                L.append(T("- Ramachandran basin 的定义：alphaR、beta、alphaL 三个中心，离中心 50° 以内算进入（这个运行在合并之前开始）。",
                           "- Ramachandran basins: centres alphaR, beta, alphaL; a frame within 50 deg of a centre is in "
                           "it (this run started before the merge)."))
            else:
                L.append(T(f"- Ramachandran basin 的定义（φ 从,到,ψ 从,到，单位度）：{bb.basins or DEFAULT_BASINS_TEXT}；"
                           f"核心区边缘留 {bb.margin_deg:g}°。β-turn 用 I 型和 II 型两个中心。",
                           f"- Ramachandran basins (phi from,to,psi from,to in degrees): {bb.basins or DEFAULT_BASINS_TEXT}; "
                           f"core margin {bb.margin_deg:g} deg. Beta turns use the centres of type I and type II."))
        perms = goal.get("permissions")
        L.append(T(f"- agent 可以自己做的事：{', '.join(perms) if perms else '无（只提建议）'}。",
                   f"- The agent could by itself: {', '.join(perms) if perms else 'nothing (suggestions only)'}."))
        L.append(T(f"- 目标：{state.get('target_description', '')}", f"- Target: {state.get('target_description', '')}"))
        L.append("")

        L += [T("## 模拟了什么", "## What was simulated"), ""]
        if setup.get("amber"):
            L.append(T(f"- 用户准备好的 Amber 体系：{', '.join(setup['amber'])}（配体保留）。",
                       f"- A prepared Amber system: {', '.join(setup['amber'])} (ligands kept)."))
        else:
            L.append(T(f"- 输入 {state.get('pdb', '')}；力场 {setup.get('forcefield', '?')}；pH {setup.get('ph', '?')}；"
                       f"{setup.get('ion_conc_molar', '?')} M {setup.get('cation', '')}Cl；水盒子 {setup.get('box_shape', '?')}，"
                       f"边距 {setup.get('padding_nm', '?')} nm。配体被去掉了，模拟的是不含配体的蛋白。",
                       f"- Input {state.get('pdb', '')}; force field {setup.get('forcefield', '?')}; pH {setup.get('ph', '?')}; "
                       f"{setup.get('ion_conc_molar', '?')} M {setup.get('cation', '')}Cl; {setup.get('box_shape', '?')} box, "
                       f"padding {setup.get('padding_nm', '?')} nm. Ligands were removed: the protein was simulated without them."))
        md = state.get("md", {})
        L.append(T(f"- 温度 {md.get('temperature_k', 300):g} K；普通 MD {state.get('n_replicas', '?')} 条重复，"
                   f"每条 {sum(state.get('rounds_ns', [])):g} ns。",
                   f"- {md.get('temperature_k', 300):g} K; plain MD with {state.get('n_replicas', '?')} replicas of "
                   f"{sum(state.get('rounds_ns', [])):g} ns each."))
        es = state.get("enhanced") or {}
        if (run_dir / "enhanced" / "populations.json").exists():
            pj = _load(run_dir / "enhanced" / "populations.json")
            bias_zh, bias_en = (("φ 和 ψ", "phi and psi") if pj.get("kind") == "backbone" else ("chi1", "chi1"))
            L.append(T(f"- 增强采样（{bias_zh} 上的 well-tempered metadynamics）{pj.get('n_replicas', '?')} 条重复，"
                       f"每条 {pj.get('ns_per_replica', '?')} ns；比例用 Tiwary–Parrinello 方法重新加权。",
                       f"- Enhanced sampling (well-tempered metadynamics on {bias_en}) with {pj.get('n_replicas', '?')} "
                       f"replicas of {pj.get('ns_per_replica', '?')} ns; populations reweighted (Tiwary-Parrinello)."))
        c = _cost(state)
        L.append(T(f"- 计算量：生产 {c.aggregate_ns:g} ns（所有重复相加），加上 {c.n_replicas} 条重复的平衡 "
                   f"{c.overhead_ns:g} ns，合计 {c.compute_ns:g} ns"
                   + (f"，实际用了 {c.measured_seconds / 3600:.1f} 小时。" if c.measured_seconds else "。"),
                   f"- Cost: {c.aggregate_ns:g} ns of production (all replicas added up) plus {c.overhead_ns:g} ns to "
                   f"equilibrate {c.n_replicas} replicas, {c.compute_ns:g} ns in total"
                   + (f", which took {c.measured_seconds / 3600:.1f} hours." if c.measured_seconds else ".")))
        L.append("")

        L += [T("## 结论", "## Conclusion"), ""]
        if not results:
            L.append(T("还没有结果。", "No results yet."))
        elif events:
            for r in results:
                seen = (r.get("extra") or {}).get("seen_ns") or []
                where = [f"{T('重复', 'replica')} {i} {T('在', 'at')} {t:g} ns" for i, t in enumerate(seen) if t is not None]
                if where:
                    L.append(T(f"- {r['label']}：看到了（{'；'.join(where)}）。",
                               f"- {r['label']}: seen ({'; '.join(where)})."))
                else:
                    L.append(T(f"- {r['label']}：没有看到（各重复的状态变化次数：{r['transitions']}）。",
                               f"- {r['label']}: not seen (state changes per replica: {r['transitions']})."))
            L += ["", T("下表是各状态所占的时间，只作参考：这类目标不要求比例的精度。",
                        "The table gives the time spent in each state, for information only: this goal asks for no "
                        "precision of populations.")]
        elif quantity == "structure_stability":
            steady = [r for r in results if r["met"]]
            if len(steady) == len(results):
                per = sum(state.get("rounds_ns", []))
                total = per * state.get("n_replicas", 0)
                goal_rate = goal.get("max_departure_rate_per_us", 10.0)
                L.append(T(f"在模拟过的时间里（每条 {per:g} ns，所有重复合计 {total:g} ns），RMSD 和 Rg 都达到了稳态，平均值"
                           f"达到了要求的精度，而且离开起始结构的速率（95% 上限）不超过每 µs {goal_rate:g} 次。在这个标准下，"
                           "结构是稳定的。它不能排除更慢的变化：想排除更慢的离开，要设更低的速率上限，并跑得更久。",
                           f"Within the time simulated ({per:g} ns per replica, {total:g} ns in all), RMSD and Rg both "
                           "reached a steady state, their means reached the requested precision, and the rate of leaving "
                           f"the start structure is at most {goal_rate:g} per µs (95 % upper bound). By this standard the "
                           "structure is stable. Slower changes are not ruled out: that needs a lower rate goal and a "
                           "longer run."))
            else:
                L.append(T(f"{', '.join(r['label'] for r in results if not r['met'])} 还没有确认达到稳态和要求的精度"
                           "（判断写在表里）。下面的平均值只能作为粗略的参考，不能作为结论。",
                           f"{', '.join(r['label'] for r in results if not r['met'])}: a steady state at the requested "
                           "precision is not confirmed (see the verdict in the table). The means below are only a rough "
                           "guide, not a conclusion."))
        elif not open_:
            L.append(T(f"全部 {len(results)} 个项目都达到了要求的精度。下面的比例可以作为这个力场和条件下的平衡比例，"
                       "引用时请附上误差。",
                       f"All {len(results)} items reached the requested precision. The populations below can be quoted "
                       "as equilibrium populations for this force field and these conditions, with their errors."))
        else:
            L.append(T(f"{len(met)} 个项目达到了要求的精度，{len(open_)} 个没有：{', '.join(r['label'] for r in open_)}。"
                       "没达到的项目，比例只能作为粗略的参考，不能作为结论。",
                       f"{len(met)} items reached the requested precision and {len(open_)} did not: "
                       f"{', '.join(r['label'] for r in open_)}. For those, the populations are only a rough guide, "
                       "not a conclusion."))
        head = (T("平均值 ± 标准误差", "Mean ± standard error") if quantity == "structure_stability"
                else T("比例 ± 标准误差", "Population ± standard error"))
        L += ["", T(f"| 项目 | {head} | 来自 | 达到要求 | 判断 |", f"| Item | {head} | From | Goal met | Verdict |"),
              "|---|---|---|---|---|"]
        for r in results:
            src = (T("增强采样（重新加权）", "enhanced sampling (reweighted)") if r["source"] == "enhanced_sampling"
                   else T("普通 MD", "plain MD"))
            L.append(f"| {r['label']} | {_fmt(r['populations'], r['standard_error'])} | {src} | "
                     f"{T('是', 'yes') if r['met'] else T('否', 'no')} | {r['verdict']} |")
        L.append("")
        L += _backbone_section(run_dir, state, T)
        L += _stability_section(run_dir, state, T)
        L += _cluster_section(run_dir, T)
        L += _decision_section(state, T)

        bb_files = sorted((run_dir / "backbone").glob("*.json")) if quantity == "backbone_basins" else []
        if bb_files:
            L += [T("## basin 之间的转变和自由能面", "## Transitions between basins and the free-energy landscape"), "",
                  T("速率的误差按转变次数的泊松误差算（假设在 basin 核心之间的跳跃是马尔可夫的；有相关时会偏小）。"
                    "平均停留时间 = 在这个 basin 的总时间 / 离开它的次数。没见过的转变，速率记为 0，误差约为一次事件。",
                    "Rate errors are Poisson errors of the counts (they assume Markovian jumps between basin cores and "
                    "are too small if visits are correlated). Mean dwell time = time in the basin / number of exits. A "
                    "transition never "
                    "seen has rate 0 and an error of about one event."), ""]
            for path in bb_files:
                d = _load(path)
                tr = d.get("transitions", {})
                L.append(f"**{d.get('residue', path.stem)}**")
                for a, row in tr.get("rates_per_ns", {}).items():
                    seen = [f"{a}→{b} {r:.3g} ± {e:.2g} /ns" for b, (r, e) in row.items() if r > 0]
                    if seen:
                        L.append("- " + "; ".join(seen))
                dw = [f"{a} {v[0]:.3g} ± {v[1]:.2g} ns ({v[2]})" for a, v in tr.get("mean_dwell_ns", {}).items() if v]
                if dw:
                    L.append(T("- 平均停留时间（离开次数）：", "- Mean dwell time (number of exits): ") + "; ".join(dw))
                mins = [f"{a} {v[0]:g}" + (f" ± {v[1]:g}" if v[1] is not None else "") + f" kcal/mol ({v[2]:g}°, {v[3]:g}°)"
                        for a, v in (d.get("basin_minima_kcal") or {}).items() if v]
                if mins:
                    L.append(T("- 各 basin 里自由能最低处（φ, ψ）：", "- Lowest free energy in each basin (phi, psi): ")
                             + "; ".join(mins))
                L.append(T(f"- 自由能面（普通 MD）：{path.with_name(path.stem + '_fes.csv').resolve()}（也有 .npz）",
                           f"- Landscape (plain MD): {path.with_name(path.stem + '_fes.csv').resolve()} (also .npz)"))
                rw = path.with_name(path.stem + "_fes_reweighted.csv")
                if rw.exists():
                    L.append(T(f"- 自由能面（增强采样，重新加权；比例也以它为准）：{rw.resolve()}（也有 .npz）",
                               f"- Landscape (enhanced sampling, reweighted; the populations above come from it): "
                               f"{rw.resolve()} (also .npz)"))
                L.append("")

        L += [T("## 为什么停下", "## Why it stopped"), ""]
        for phase, o in outcomes:
            if not o:
                continue
            name = {"plain_md": T("普通 MD", "Plain MD"), "enhanced_sampling": T("增强采样", "Enhanced sampling"),
                    "any": T("运行", "The run")}[phase]
            why = REASONS.get(o.get("reason"), (o.get("reason"), o.get("reason")))[0 if zh else 1]
            extra = (T(f"，每条 {o['ns_per_replica']:g} ns（预算 {o['budget_ns']:g} ns）",
                       f", after {o['ns_per_replica']:g} ns per replica (budget {o['budget_ns']:g} ns)")
                     if "ns_per_replica" in o and "budget_ns" in o else "")
            err = f"：{o['error']}" if zh and o.get("error") else (f": {o['error']}" if o.get("error") else "")
            L.append(f"- {name}：{why}{extra}{err}" if zh else f"- {name}: {why}{extra}{err}")
        actions = notes.get("actions", [])
        if actions:
            L += ["", T("agent 采取或建议过的动作：", "Actions the agent took or suggested:"), ""]
            for a in actions:
                done = T("已执行", "done") if a["executed"] else T("只建议", "suggested")
                L.append(f"- {a['time']} {a['phase']} {a['action']} ({done}): {a['reason']}")
        L.append("")

        L += [T("## 可信度和限制", "## Trust and limits"), ""]
        L.append(T("- 误差考虑了时间相关（分块平均）和重复之间的差别。这些检查能发现采样不足，但不能证明采样充分。",
                   "- Errors allow for time correlation (block averaging) and the spread between replicas. These checks "
                   "can show poor sampling but cannot prove good sampling."))
        L.append(T("- 结果只对所用的力场、温度和体系成立。", "- The result holds for the force field, temperature and system used."))
        frozen = [r["label"] for r in results if r["transitions"] and sum(r["transitions"]) == 0]
        if frozen:
            L.append(T(f"- {', '.join(frozen)} 从来没有改变状态。它们被困住了；误差只按“看到得太少”估计（误差下限），"
                       "比例只说明它们一直停在哪个状态。",
                       f"- {', '.join(frozen)} never changed state: they are trapped. Their error only reflects how "
                       "little was seen (error floor); the population only says which state they stayed in."))
        if any(r["verdict"] == "stalled" for r in results):
            L.append(T("- 判为 stalled 的项目在状态之间几乎没有来回，它们的比例不能信，即使各条重复看起来一致。",
                       "- Items judged stalled hardly moved between states; their populations cannot be trusted even "
                       "if the replicas look alike."))
        if any(r["source"] == "enhanced_sampling" for r in results):
            if quantity == "backbone_basins":
                L.append(T("- 增强采样只偏置了 φ 和 ψ（每个二面角一个一维偏置势）。和别的慢运动（例如侧链或整体构象）耦合在一起的残基，"
                           "可能仍然采样不足。偏置会改变动力学，所以转变速率和停留时间只来自普通 MD；basin 比例和自由能面来自重新加权。",
                           "- Enhanced sampling biased phi and psi only (one one-dimensional bias per dihedral). Residues "
                           "coupled to other slow motions may still be under-sampled. The bias changes the kinetics, so "
                           "transition rates and dwell times come from plain MD only; basin populations and the landscape "
                           "come from reweighting."))
            else:
                L.append(T("- 增强采样只偏置了 chi1。和别的慢运动耦合在一起的侧链，可能仍然采样不足。",
                           "- Enhanced sampling biased chi1 only. Side chains coupled to other slow motions may still be "
                           "under-sampled."))
        L.append("")

        L += [T("## 文件", "## Files"), "", "```", *files,
              f"  {'report.zh.md, report.en.md':<36} " + T("这份报告", "this report"), "```", ""]

        L += [T("## 还剩的问题", "## Open questions"), ""]
        if not open_:
            L.append(T("没有。", "None."))
        for r in open_:
            hint = {"stalled": T("采样停滞：试别的方法，或者偏置和它耦合的运动。", "stalled: try another method, or bias the motion it is coupled to."),
                    "not_yet": T("还不够精确：加长或加重复。", "not precise yet: run longer or add replicas."),
                    "unreliable": T("权重退化：加长增强采样。", "degenerate weights: run enhanced sampling longer."),
                    "imprecise": T("还不够精确：加长或加重复。", "not precise yet: run longer or add replicas."),
                    "too_short": T("观察时间还不够长，离开速率的上限还没达到目标：加长。",
                                   "watched too briefly for the departure-rate goal: run longer.")}.get(
                r["verdict"], T("没有采样充分：加长、加重复或换方法。", "not sampled well: run longer, add replicas or change method."))
            L.append(f"- {r['label']}：{hint}" if zh else f"- {r['label']}: {hint}")
        path = run_dir / f"report.{lang}.md"
        path.write_text("\n".join(L) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def equivalent(pa: float, ea: float, pb: float, eb: float, tolerance: float,
               alpha: float = 0.05) -> tuple[float, bool]:
    """Are two populations the same within `tolerance`? Returns the difference and the verdict.

    Equivalent when the whole (1 - 2*alpha) interval of the difference lies inside
    (-tolerance, +tolerance), which is the usual two one-sided test (TOST). An imprecise
    result is therefore never called equivalent, which is what a short run has to prove.
    """
    d = pa - pb
    err = math.sqrt((ea if math.isfinite(ea) else 0.0) ** 2 + (eb if math.isfinite(eb) else 0.0) ** 2)
    if not math.isfinite(d):
        return float("nan"), False
    z = NormalDist().inv_cdf(1 - alpha)
    return d, abs(d) + z * err <= tolerance


def compare(run: Path, reference: Path, tolerance: float | None = None,
            alpha: float = 0.05) -> tuple[str, bool]:
    """Compare the final results of two runs. Returns a table and whether they are equivalent.

    `tolerance` defaults to the precision the compared run asked for. Without any
    tolerance (an old run with no goal), it falls back to 0.05.
    """
    a = {r["label"]: r for r in final_results(run)}
    b = {r["label"]: r for r in final_results(reference)}
    if tolerance is None:
        tolerance = ((_load(Path(run) / "run.json").get("goal") or {}).get("precision")) or 0.05
    common = [k for k in a if k in b]
    lines = [f"Tolerance: {tolerance:g} (populations equal within this count as the same answer), "
             f"confidence {1 - 2 * alpha:.0%}.",
             f"{'item':<24}{'state':<8}{'run':>14}{'reference':>16}{'diff':>8}{'z':>8}{'same':>7}"]
    agree = bool(common)
    for k in common:
        for s in a[k]["populations"]:
            pa, pb = a[k]["populations"][s], b[k]["populations"].get(s, float("nan"))
            ea, eb = a[k]["standard_error"][s], b[k]["standard_error"].get(s, float("nan"))
            err = math.sqrt(ea ** 2 + eb ** 2)
            z = (pa - pb) / err if err > 0 else (0.0 if pa == pb else float("inf"))
            d, same = equivalent(pa, ea, pb, eb, tolerance, alpha)
            agree = agree and same
            lines.append(f"{k:<24}{s:<8}{pa:>8.3f}±{ea:<5.3f}{pb:>10.3f}±{eb:<5.3f}"
                         f"{d:>8.3f}{z:>8.2f}{'yes' if same else 'no':>7}")
    missing = sorted(set(a) ^ set(b))
    if missing:
        lines.append(f"Only in one run: {', '.join(missing)}")
    lines.append(f"The two runs give the same answer within {tolerance:g} for every state." if agree else
                 f"The two runs are NOT shown to give the same answer within {tolerance:g} (the difference or "
                 "the error is too large), or they have no item in common.")
    return "\n".join(lines), agree

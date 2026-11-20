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

from .outputs import describe, host

REASONS = {
    "target_reached": ("达到目标", "goal reached"),
    "stalled": ("采样停滞", "sampling stalled"),
    "budget_exhausted": ("预算用完", "budget used up"),
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


def _precision(state: dict):
    """The precision the run asked for, read back from run.json."""
    from .uncertainty import Precision, planned_looks

    goal, loop = state.get("goal") or {}, state.get("loop") or {}
    factor = goal.get("extend_factor", 2.0) if "extend" in (goal.get("permissions") or []) else 1.0
    return Precision(epsilon=goal.get("precision", 0.05), criterion=goal.get("criterion", "standard_error"),
                     alpha=goal.get("alpha", 0.05),
                     looks=planned_looks(loop.get("first_ns", 1.0), loop.get("extend_ns", 5.0),
                                         loop.get("max_ns", 20.0), factor))


def _fmt(pops: dict, errs: dict) -> str:
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
    files = describe(run_dir)
    paths = []
    for lang in ("zh", "en"):
        zh = lang == "zh"

        def T(a, b):
            return a if zh else b
        L = [T(f"# MDAgent 结果报告：{run_dir.name}", f"# MDAgent result report: {run_dir.name}"), "",
             T(f"写于 {time.strftime('%Y-%m-%d %H:%M')}，电脑“{host()}”，文件夹 {run_dir.resolve()}。",
               f"Written {time.strftime('%Y-%m-%d %H:%M')} on computer '{host()}', folder {run_dir.resolve()}."), ""]

        L += [T("## 问题", "## The question"), ""]
        quantity = goal.get("quantity", "chi1_populations")
        what = (T("原子间距离处在“近”和“远”两种状态的比例", "populations of the near and far states of atom distances")
                if quantity == "distance_states" else
                T("目标区域侧链的 chi1 构象（g+、t、g-）的比例", "populations of the chi1 states (g+, t, g-) of the target side chains"))
        L.append(T(f"- 目标量：{what}。", f"- Quantity: {what}."))
        if goal.get("description"):
            L.append(T(f"- 用户的原话：{goal['description']}", f"- In the user's words: {goal['description']}"))
        L.append(T(f"- 要求的精度：{_precision(state).describe(zh=True)}。",
                   f"- Requested precision: {_precision(state).describe()}."))
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
            L.append(T(f"- 增强采样（chi1 上的 well-tempered metadynamics）{pj.get('n_replicas', '?')} 条重复，"
                       f"每条 {pj.get('ns_per_replica', '?')} ns；比例用 Tiwary–Parrinello 方法重新加权。",
                       f"- Enhanced sampling (well-tempered metadynamics on chi1) with {pj.get('n_replicas', '?')} "
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
        L += ["", T("| 项目 | 比例 ± 标准误差 | 来自 | 达到要求 | 判断 |", "| Item | Population ± standard error | From | Goal met | Verdict |"),
              "|---|---|---|---|---|"]
        for r in results:
            src = (T("增强采样（重新加权）", "enhanced sampling (reweighted)") if r["source"] == "enhanced_sampling"
                   else T("普通 MD", "plain MD"))
            L.append(f"| {r['label']} | {_fmt(r['populations'], r['standard_error'])} | {src} | "
                     f"{T('是', 'yes') if r['met'] else T('否', 'no')} | {r['verdict']} |")
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
            L.append(T(f"- {', '.join(frozen)} 从来没有改变状态。它们的误差显示为 0，这不表示很精确，而是表示被困住了；"
                       "比例只说明它们一直停在哪个状态。",
                       f"- {', '.join(frozen)} never changed state. Their error shows as 0; that does not mean precise, "
                       "it means trapped. The population only says which state they stayed in."))
        if any(r["verdict"] == "stalled" for r in results):
            L.append(T("- 判为 stalled 的项目在状态之间几乎没有来回，它们的比例不能信，即使各条重复看起来一致。",
                       "- Items judged stalled hardly moved between states; their populations cannot be trusted even "
                       "if the replicas look alike."))
        if any(r["source"] == "enhanced_sampling" for r in results):
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
                    "imprecise": T("还不够精确：加长或加重复。", "not precise yet: run longer or add replicas.")}.get(
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

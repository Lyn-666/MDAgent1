"""Find side chains in the target region that were not sampled well.

For each target residue, we look at the side-chain angle chi1 and assign
each frame to one of three rotamer states: g+ (near +60°), t (near 180°)
and g- (near -60°). A frame counts as "in a state" only when the angle is
within `core_width_deg` of the state's center. Between states, the frame
keeps the last state it was in. This avoids counting thermal noise near a
boundary as a flip.

Verdicts for each residue:
- "no_flip": chi1 never changed state in any replica. The side chain may be
  trapped, or it may have only one stable state. Plain MD cannot tell
  these two cases apart. Enhanced sampling can.
- "replicas_disagree": replicas spent most of their time in different states.
  The result depends on which replica you look at, so it is not converged.
- "rare_flips": chi1 changed state, but too few times to estimate the
  population of each state.
- "ok": chi1 changed state often enough in every replica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mdtraj as md
import numpy as np

from .target import ResidueKey

STATE_CENTERS_DEG = {"g+": 60.0, "t": 180.0, "g-": -60.0}
STATE_NAMES = list(STATE_CENTERS_DEG)


def angular_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def assign_states(angles_deg: np.ndarray, core_width_deg: float = 40.0) -> np.ndarray:
    """Return the state index of each frame, or -1 when the frame is between states.

    Each frame first gets the state of the core it is in (or -1). Then frames
    outside every core keep the last state visited.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    core = np.full(angles_deg.shape, -1, dtype=int)
    for i, center in enumerate(STATE_CENTERS_DEG.values()):
        core[angular_distance_deg(angles_deg, center) <= core_width_deg] = i
    states = core.copy()
    for t in range(1, len(states)):
        if states[t] == -1:
            states[t] = states[t - 1]
    return states


def count_transitions(states: np.ndarray) -> int:
    s = states[states >= 0]
    return int(np.count_nonzero(s[1:] != s[:-1]))


def occupancy(states: np.ndarray) -> dict[str, float]:
    s = states[states >= 0]
    if len(s) == 0:
        return {name: 0.0 for name in STATE_NAMES}
    return {name: float(np.mean(s == i)) for i, name in enumerate(STATE_NAMES)}


@dataclass
class ResidueReport:
    chain: str
    resseq: int
    resname: str
    transitions: list[int]
    occupancies: list[dict[str, float]]
    dominant: list[str]
    verdict: str = ""
    reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.resname}{self.resseq} (chain {self.chain})"


def judge(report: ResidueReport, min_flips_per_replica: int = 4) -> None:
    total = sum(report.transitions)
    n = len(report.transitions)
    if total == 0:
        report.verdict = "no_flip"
        report.reason = (f"chi1 stayed in state {report.dominant[0]} in all {n} replicas. "
                         "It may be trapped, or it may have only one stable state.")
    elif len(set(report.dominant)) > 1:
        report.verdict = "replicas_disagree"
        report.reason = ("Replicas spend most time in different states: "
                         + ", ".join(report.dominant) + ".")
    elif min(report.transitions) < min_flips_per_replica:
        report.verdict = "rare_flips"
        report.reason = (f"chi1 flips per replica: {report.transitions}. "
                         f"Fewer than {min_flips_per_replica} in at least one replica.")
    else:
        report.verdict = "ok"
        report.reason = f"chi1 flips per replica: {report.transitions}."


def diagnose_trajectories(
    trajs: list[md.Trajectory],
    target: list[ResidueKey],
    core_width_deg: float = 40.0,
    min_flips_per_replica: int = 4,
) -> list[ResidueReport]:
    wanted = set(target)
    top = trajs[0].topology
    chi_idx, _ = md.compute_chi1(trajs[0][:1])
    keep = []
    for k, quad in enumerate(chi_idx):
        res = top.atom(int(quad[1])).residue
        if (res.chain.chain_id, res.resSeq) in wanted:
            keep.append((k, res))

    per_traj_angles = [np.degrees(md.compute_chi1(t)[1]) for t in trajs]
    reports = []
    for k, res in keep:
        states = [assign_states(a[:, k], core_width_deg) for a in per_traj_angles]
        occ = [occupancy(s) for s in states]
        rep = ResidueReport(
            chain=res.chain.chain_id,
            resseq=res.resSeq,
            resname=res.name,
            transitions=[count_transitions(s) for s in states],
            occupancies=occ,
            dominant=[max(o, key=o.get) for o in occ],
        )
        judge(rep, min_flips_per_replica)
        reports.append(rep)

    no_chi = sorted(wanted - {(r.chain, r.resseq) for r in reports})
    if no_chi:
        names = {(r.chain.chain_id, r.resSeq): r.name for r in top.residues}
        for key in no_chi:
            reports.append(ResidueReport(
                chain=key[0], resseq=key[1], resname=names.get(key, "?"),
                transitions=[], occupancies=[], dominant=[],
                verdict="no_chi1", reason="Residue has no chi1 angle (Gly or Ala).",
            ))
    reports.sort(key=lambda r: (r.chain, r.resseq))
    return reports


def load_run(run_dir: str | Path) -> list[md.Trajectory]:
    run_dir = Path(run_dir)
    top = str(run_dir / "protein.pdb")
    dcds = sorted(run_dir.glob("replica_*.dcd"))
    if not dcds:
        raise FileNotFoundError(f"No replica_*.dcd files in {run_dir}.")
    return [md.load_dcd(str(p), top=top) for p in dcds]


VERDICT_ORDER = {"replicas_disagree": 0, "no_flip": 1, "rare_flips": 2, "ok": 3, "no_chi1": 4}


def format_report(reports: list[ResidueReport], ns_per_replica: float | None = None) -> str:
    lines = []
    flagged = [r for r in reports if r.verdict in ("replicas_disagree", "no_flip", "rare_flips")]
    length = f" in {ns_per_replica:g} ns per replica" if ns_per_replica else ""
    lines.append(f"{len(flagged)} of {len(reports)} target residues were not sampled well{length}.")
    lines.append("")
    lines.append(f"{'residue':<20}{'verdict':<20}{'flips per replica':<20}dominant state per replica")
    for r in sorted(reports, key=lambda r: (VERDICT_ORDER[r.verdict], r.resseq)):
        flips = ",".join(map(str, r.transitions)) or "-"
        dom = ",".join(r.dominant) or "-"
        lines.append(f"{r.label:<20}{r.verdict:<20}{flips:<20}{dom}")
    if flagged:
        lines.append("")
        lines.append("Candidates for enhanced sampling of chi1:")
        for r in flagged:
            lines.append(f"- {r.label}: {r.reason}")
    return "\n".join(lines)

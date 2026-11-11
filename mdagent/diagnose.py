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

Ranking: residues are ranked by the estimated time between two flips.
A slow side chain is ranked first. With zero flips, we only know a lower
bound: the total simulated time. Ties are broken by how much the replicas
disagree about the state populations.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np

from .target import ResidueKey

STATE_CENTERS_DEG = {"g+": 60.0, "t": 180.0, "g-": -60.0}
STATE_NAMES = list(STATE_CENTERS_DEG)
NEEDS_WORK = ("replicas_disagree", "no_flip", "rare_flips")


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
    ns_per_replica: float = 0.0
    verdict: str = ""
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.resname}{self.resseq}"

    @property
    def total_ns(self) -> float:
        return self.ns_per_replica * len(self.transitions)

    @property
    def time_between_flips_ns(self) -> float:
        """Estimated mean time between flips. With zero flips this is a lower bound."""
        flips = sum(self.transitions)
        return self.total_ns / flips if flips else self.total_ns

    @property
    def disagreement(self) -> float:
        """Largest difference between replicas in the population of any state (0 to 1)."""
        if len(self.occupancies) < 2:
            return 0.0
        return max(
            max(o[n] for o in self.occupancies) - min(o[n] for o in self.occupancies)
            for n in STATE_NAMES
        )

    def time_between_flips_text(self) -> str:
        if not self.transitions:
            return "-"
        prefix = ">" if sum(self.transitions) == 0 else "~"
        t = self.time_between_flips_ns
        return f"{prefix}{t:.1f} ns" if t >= 0.1 else f"{prefix}{t * 1000:.0f} ps"


def judge(report: ResidueReport, min_flips_per_replica: int = 4) -> None:
    total = sum(report.transitions)
    n = len(report.transitions)
    if len(set(report.dominant)) > 1:
        # Checked first: replicas can end up in different states with zero flips
        # during production, if they separated during equilibration.
        report.verdict = "replicas_disagree"
        report.reason = (f"Replicas spend most time in different states ({', '.join(report.dominant)}). "
                         f"chi1 flips per replica: {report.transitions}.")
    elif total == 0:
        report.verdict = "no_flip"
        report.reason = (f"chi1 stayed in state {report.dominant[0]} in all {n} replicas. "
                         "It may be trapped, or it may have only one stable state.")
    elif min(report.transitions) < min_flips_per_replica:
        report.verdict = "rare_flips"
        report.reason = (f"chi1 flips per replica: {report.transitions}. "
                         f"Fewer than {min_flips_per_replica} in at least one replica.")
    else:
        report.verdict = "ok"
        report.reason = f"chi1 flips per replica: {report.transitions}."


def rank(reports: list[ResidueReport]) -> list[ResidueReport]:
    """Residues that need work, slowest first. Then the others."""
    def key(r: ResidueReport):
        needs = r.verdict in NEEDS_WORK
        return (not needs, -r.time_between_flips_ns, -r.disagreement, r.resseq)
    return sorted(reports, key=key)


def diagnose_trajectories(
    trajs: list[md.Trajectory],
    target: list[ResidueKey],
    ns_per_replica: float,
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
            ns_per_replica=ns_per_replica,
        )
        judge(rep, min_flips_per_replica)
        reports.append(rep)

    names = {(r.chain.chain_id, r.resSeq): r.name for r in top.residues}
    for key in sorted(wanted - {(r.chain, r.resseq) for r in reports}):
        reports.append(ResidueReport(
            chain=key[0], resseq=key[1], resname=names.get(key, "?"),
            transitions=[], occupancies=[], dominant=[],
            verdict="no_chi1", reason="Residue has no chi1 angle (Gly or Ala).",
        ))
    return rank(reports)


@contextlib.contextmanager
def _quiet_c_output():
    """Hide messages that the DCD reader prints from C code.

    C code keeps its messages in a buffer. We flush that buffer while the
    output still goes to /dev/null. Otherwise the messages appear when the
    program exits.
    """
    import ctypes

    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)
        try:
            yield
        finally:
            libc.fflush(None)
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def load_run(run_dir: str | Path, target: list[ResidueKey], n_parts: int) -> list[md.Trajectory]:
    """Load parts 0 .. n_parts-1 of each replica and join them into one trajectory per replica.

    Only the atoms of the target residues are kept, so long runs fit in memory.
    """
    run_dir = Path(run_dir)
    top = md.load_topology(str(run_dir / "protein.pdb"))
    wanted = set(target)
    atoms = [a.index for a in top.atoms if (a.residue.chain.chain_id, a.residue.resSeq) in wanted]
    replicas = sorted({int(p.name.split("_")[1]) for p in run_dir.glob("replica_*_part000.dcd")})
    if not replicas:
        raise FileNotFoundError(f"No trajectories in {run_dir}.")
    trajs = []
    with _quiet_c_output():
        for i in replicas:
            parts = [run_dir / f"replica_{i}_part{k:03d}.dcd" for k in range(n_parts)]
            trajs.append(md.join([md.load_dcd(str(p), top=top, atom_indices=atoms) for p in parts]))
    return trajs


def format_report(reports: list[ResidueReport]) -> str:
    flagged = [r for r in reports if r.verdict in NEEDS_WORK]
    ns = next((r.ns_per_replica for r in reports if r.transitions), 0.0)
    lines = [
        f"{len(flagged)} of {len(reports)} target residues were not sampled well "
        f"after {ns:g} ns per replica.",
        "Rank 1 is the slowest side chain.",
        "",
        f"{'rank':<6}{'residue':<10}{'verdict':<20}{'flips per replica':<20}"
        f"{'time between flips':<20}dominant state per replica",
    ]
    for i, r in enumerate(reports, 1):
        rank_text = str(i) if r.verdict in NEEDS_WORK else "-"
        flips = ",".join(map(str, r.transitions)) or "-"
        dom = ",".join(r.dominant) or "-"
        lines.append(f"{rank_text:<6}{r.label:<10}{r.verdict:<20}{flips:<20}"
                     f"{r.time_between_flips_text():<20}{dom}")
    return "\n".join(lines)

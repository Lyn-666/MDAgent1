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
- "imprecise": chi1 flipped often enough, but the error of a state population
  (block averaging and spread between replicas, see uncertainty.py) is larger
  than the precision the user asked for.
- "ok": chi1 changed state often enough in every replica, and the populations
  are as precise as the user asked for.
- "fixed_chi1": chi1 cannot change rotamer state, so it is not judged and
  gets no enhanced sampling. This is proline (the ring holds chi1) and
  cysteine in a disulfide bond (the S-S bond holds chi1).

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
from .uncertainty import block_standard_error, combine_replicas, error_parts, floor_from_transitions

STATE_CENTERS_DEG = {"g+": 60.0, "t": 180.0, "g-": -60.0}
STATE_NAMES = list(STATE_CENTERS_DEG)
NEEDS_WORK = ("replicas_disagree", "no_flip", "rare_flips", "imprecise", "not_yet")


DISULFIDE_MAX_NM = 0.25  # SG-SG distance; a disulfide bond is about 0.205 nm.


def _is_proline(resname: str) -> bool:
    return resname in ("PRO", "NPRO", "CPRO")


def fixed_chi1_residues(protein_pdb: str | Path) -> dict[ResidueKey, str]:
    """Residues whose chi1 cannot change rotamer state, with the reason.

    Proline: the ring holds chi1. Cysteine in a disulfide bond: the S-S bond holds
    chi1. Disulfides are found by the SG-SG distance, so this works for every
    force field and residue naming (CYS, CYX).
    """
    t = md.load(str(protein_pdb))
    top = t.topology
    out = {}
    for r in top.residues:
        if _is_proline(r.name):
            out[(r.chain.chain_id, r.resSeq)] = "Proline: the ring holds chi1, so it has no rotamer states to sample."
    sg = [a for a in top.atoms if a.name == "SG"]
    if len(sg) >= 2:
        pairs = np.array([(a.index, b.index) for i, a in enumerate(sg) for b in sg[i + 1:]])
        for (i, j), d in zip(pairs, md.compute_distances(t, pairs)[0]):
            if d < DISULFIDE_MAX_NM:
                a, b = top.atom(int(i)).residue, top.atom(int(j)).residue
                for x, y in ((a, b), (b, a)):
                    out[(x.chain.chain_id, x.resSeq)] = (
                        f"Cysteine in a disulfide bond with {y.name}{y.resSeq}: the S-S bond holds chi1.")
    return out


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


def occupancy(states: np.ndarray, names: list[str] = STATE_NAMES) -> dict[str, float]:
    s = states[states >= 0]
    if len(s) == 0:
        return {name: 0.0 for name in names}
    return {name: float(np.mean(s == i)) for i, name in enumerate(names)}


def populations_with_errors(states: list[np.ndarray], names: list[str]) -> tuple[dict, dict, bool, bool]:
    """Mean population of each state over replicas, its error (time correlation and replica spread),
    whether two replicas differ beyond their errors, and whether the largest error comes mainly from
    the spread between replicas (then more replicas help more than longer ones)."""
    pops, errs, differ, worst, from_replicas = {}, {}, False, -1.0, False
    changes = sum(count_transitions(s_) for s_ in states)
    for i, name in enumerate(names):
        hits = [(s_[s_ >= 0] == i).astype(float) for s_ in states]
        values = [float(h.mean()) if h.size else 0.0 for h in hits]
        errors = [block_standard_error(h) for h in hits]
        mean, err, dis = combine_replicas(values, errors)
        err = max(err, floor_from_transitions(mean, changes, len(states)))  # Never 0 for an unseen state.
        pops[name], errs[name], differ = round(mean, 3), round(err, 3), differ or dis
        if err > worst:
            within, between = error_parts(values, errors)
            worst, from_replicas = err, between > within
    return pops, errs, differ, from_replicas


@dataclass
class ResidueReport:
    chain: str
    resseq: int
    resname: str
    transitions: list[int]
    occupancies: list[dict[str, float]]
    dominant: list[str]
    ns_per_replica: float = 0.0
    total_ns_override: float | None = None  # Adaptive mode: segments have different lengths.
    verdict: str = ""
    reason: str = ""
    populations: dict | None = None   # Mean over replicas.
    standard_error: dict | None = None  # Of each population, allowing for time correlation.
    name: str = ""            # Label of an item that is not a residue (for example a distance).
    quantity: str = "chi1"    # What has the states: "chi1" or "distance" (observables.py).
    extra: dict | None = None  # Other numbers worth showing, for example the mean distance.
    replicas_differ: bool | None = None  # Replica populations differ beyond their errors (None: not computed).
    error_from_replicas: bool = False    # The largest error comes mainly from the spread between replicas.

    @property
    def label(self) -> str:
        return self.name or f"{self.resname}{self.resseq}"

    @property
    def total_ns(self) -> float:
        if self.total_ns_override is not None:
            return self.total_ns_override
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
            for n in self.occupancies[0]
        )

    def time_between_flips_text(self) -> str:
        if not self.transitions:
            return "-"
        prefix = ">" if sum(self.transitions) == 0 else "~"
        t = self.time_between_flips_ns
        return f"{prefix}{t:.1f} ns" if t >= 0.1 else f"{prefix}{t * 1000:.0f} ps"


def judge(report: ResidueReport, min_flips_per_replica: int = 4, precision: float | None = None) -> None:
    total = sum(report.transitions)
    n = len(report.transitions)
    q = "chi1" if report.quantity == "chi1" else "The distance"
    if len(set(report.dominant)) > 1 and (total == 0 or report.replicas_differ is not False):
        # Checked first: replicas can end up in different states with zero flips
        # during production, if they separated during equilibration. With flips, a different
        # most-populated state counts only if the populations differ beyond their errors:
        # with two states near 50/50 the most-populated one differs by chance.
        report.verdict = "replicas_disagree"
        report.reason = (f"Replicas spend most time in different states ({', '.join(report.dominant)}). "
                         f"{q} state changes per replica: {report.transitions}.")
    elif total == 0:
        report.verdict = "no_flip"
        report.reason = (f"{q} stayed in state {report.dominant[0]} in all {n} replicas. "
                         "It may be trapped, or it may have only one stable state.")
    elif min(report.transitions) < min_flips_per_replica:
        report.verdict = "rare_flips"
        report.reason = (f"{q} state changes per replica: {report.transitions}. "
                         f"Fewer than {min_flips_per_replica} in at least one replica.")
    elif precision is not None and report.standard_error and max(report.standard_error.values()) > precision:
        report.verdict = "imprecise"
        report.reason = (f"{q} state changes per replica: {report.transitions}, but the largest population error is "
                         f"{max(report.standard_error.values()):.3f}, more than the requested {precision:g}.")
    else:
        report.verdict = "ok"
        report.reason = f"{q} state changes per replica: {report.transitions}."


def judge_pooled(report: ResidueReport, min_total_flips: int) -> None:
    """Adaptive mode: trajectory segments start from chosen frames, so their states
    differ on purpose. Only the total number of flips over all segments counts."""
    total = sum(report.transitions)
    states = sorted(set(report.dominant))
    if total == 0 and len(states) > 1:
        report.verdict = "no_flip"
        report.reason = (f"chi1 was seen in more than one state ({', '.join(states)}), but it did not flip "
                         "inside any trajectory segment. The different states come from where the segments "
                         "started, so plain MD did not show how often chi1 changes state.")
    elif total == 0:
        report.verdict = "no_flip"
        report.reason = ("chi1 did not flip in any trajectory segment. It may be trapped, "
                         "or it may have only one stable state.")
    elif total < min_total_flips:
        report.verdict = "rare_flips"
        report.reason = f"chi1 flipped {total} times in all segments together; fewer than {min_total_flips}."
    else:
        report.verdict = "ok"
        report.reason = f"chi1 flipped {total} times in all segments together."


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
    pooled_total_ns: float | None = None,
    n_replicas: int | None = None,
    fixed: dict[ResidueKey, str] | None = None,
    precision: float | None = None,
) -> list[ResidueReport]:
    """Diagnose the target side chains.

    Default: each trajectory is one replica. With pooled_total_ns (adaptive
    mode), each trajectory is one segment, and the verdict uses the total
    number of flips (judge_pooled) over pooled_total_ns of simulation.

    fixed: residues whose chi1 cannot change state (see fixed_chi1_residues).
    They get the verdict "fixed_chi1". Prolines are found by name even without it.
    """
    wanted = set(target)
    top = trajs[0].topology
    chi_idx, _ = md.compute_chi1(trajs[0][:1])
    keep = []
    for k, quad in enumerate(chi_idx):
        res = top.atom(int(quad[1])).residue
        key = (res.chain.chain_id, res.resSeq)
        if key in wanted and _is_proline(res.name):
            fixed = {key: "Proline: the ring holds chi1, so it has no rotamer states to sample.", **(fixed or {})}
        if key in wanted and key not in (fixed or {}):
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
            total_ns_override=pooled_total_ns,
        )
        if pooled_total_ns is None:
            (rep.populations, rep.standard_error, rep.replicas_differ,
             rep.error_from_replicas) = populations_with_errors(states, STATE_NAMES)
            judge(rep, min_flips_per_replica, precision)
        else:
            judge_pooled(rep, min_flips_per_replica * (n_replicas or len(trajs)))
        reports.append(rep)

    names = {(r.chain.chain_id, r.resSeq): r.name for r in top.residues}
    for key in sorted(wanted & set(fixed or {})):
        reports.append(ResidueReport(
            chain=key[0], resseq=key[1], resname=names.get(key, "?"),
            transitions=[], occupancies=[], dominant=[], verdict="fixed_chi1", reason=fixed[key]))
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


def load_run(run_dir: str | Path, target: list[ResidueKey], n_parts: int,
             join: bool = True, atoms: list[int] | None = None) -> list[md.Trajectory]:
    """Load parts 0 .. n_parts-1 of each replica and join them into one trajectory per replica.

    Only the atoms of the target residues are kept, so long runs fit in memory. With `atoms`
    (indices in protein.pdb), exactly those atoms are kept, in that order.
    With join=False, return every part as its own trajectory: replica 0 parts,
    then replica 1 parts, and so on.
    """
    run_dir = Path(run_dir)
    top = md.load_topology(str(run_dir / "protein.pdb"))
    wanted = set(target)
    if atoms is None:
        atoms = [a.index for a in top.atoms if (a.residue.chain.chain_id, a.residue.resSeq) in wanted]
    replicas = sorted({int(p.name.split("_")[1]) for p in run_dir.glob("replica_*_part000.dcd")})
    if not replicas:
        raise FileNotFoundError(f"No trajectories in {run_dir}.")
    trajs = []
    with _quiet_c_output():
        for i in replicas:
            parts = [run_dir / f"replica_{i}_part{k:03d}.dcd" for k in range(n_parts)]
            loaded = [md.load_dcd(str(p), top=top, atom_indices=atoms) for p in parts]
            if join:
                trajs.append(md.join(loaded))
            else:
                trajs.extend(loaded)  # One trajectory per part (adaptive mode).
    return trajs


def format_report(reports: list[ResidueReport]) -> str:
    flagged = [r for r in reports if r.verdict in NEEDS_WORK]
    ns = next((r.ns_per_replica for r in reports if r.transitions), 0.0)
    distance = bool(reports) and all(r.quantity == "distance" for r in reports)
    what, item = ("distances", "distance") if distance else ("residues", "residue")
    width = max([len(item) + 2] + [len(r.label) + 2 for r in reports])
    lines = [
        f"{len(flagged)} of {len(reports)} target {what} were not sampled well "
        f"after {ns:g} ns per replica.",
        "Rank 1 is the slowest distance." if distance else "Rank 1 is the slowest side chain.",
        "",
        f"{'rank':<6}{item:<{width}}{'verdict':<20}{'changes per traj.' if distance else 'flips per traj.':<20}"
        f"{'time between changes' if distance else 'time between flips':<22}{'max error':<11}"
        "dominant state per replica",
    ]
    for i, r in enumerate(reports, 1):
        rank_text = str(i) if r.verdict in NEEDS_WORK else "-"
        flips = ",".join(map(str, r.transitions)) or "-"
        dom = ",".join(r.dominant) or "-"
        err = f"{max(r.standard_error.values()):.3f}" if r.standard_error else "-"
        lines.append(f"{rank_text:<6}{r.label:<{width}}{r.verdict:<20}{flips:<20}"
                     f"{r.time_between_flips_text():<22}{err:<11}{dom}")
    return "\n".join(lines)

"""What the user wants to know about the protein (the "quantity"), measured from trajectories.

Every quantity is turned into states, frame by frame, so that the same machinery
judges all of them: count the changes between states, compute the population of
each state with its error (uncertainty.py), and decide whether the result is good
enough (diagnose.judge, agent.decide).

Supported quantities:
- "chi1" (goal quantity "chi1_populations"): the side-chain angle chi1 of each target
  residue, with three states g+, t, g- (diagnose.py). Enhanced sampling can push it.
- "distance" (goal quantity "distance_states"): the distance between two atoms, with two
  states, "near" (below a cutoff) and "far". Good for a pocket that opens and closes,
  a contact that forms and breaks, or a gate. No enhanced sampling for it yet.

A distance is written as "A:88:OH-A:111:CG1@0.6": chain, residue number and atom name
of each atom, then the cutoff in nm. The chain may be left out ("88:OH-111:CG1@0.6"),
for example for Amber files, which have no chain names.

A third kind of goal is not a population but an event (goal quantity "event_seen"): "stop once
this change is seen". It is written as item=state, for example "A:111=t" (the chi1 of residue
111 enters t) or "A:88:OH-A:111:CG1@0.6=far" (the distance becomes far). It counts as seen when
the item enters that state from another state and stays in it for a hold time (default 100 ps),
so a short flicker or the starting state does not count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mdtraj as md
import numpy as np

from .diagnose import ResidueReport, count_transitions, judge, occupancy, populations_with_errors, rank

QUANTITIES = {"chi1": "chi1_populations", "distance": "distance_states", "event": "event_seen"}
DISTANCE_STATES = ["near", "far"]
_ATOM = r"(?:([A-Za-z0-9]):)?(-?\d+):([A-Za-z0-9']+)"
_SPEC = re.compile(rf"^\s*{_ATOM}\s*-\s*{_ATOM}\s*@\s*([0-9.]+)\s*$")


@dataclass
class DistanceSpec:
    a: tuple            # (chain or None, residue number, atom name)
    b: tuple
    cutoff_nm: float
    buffer_nm: float = 0.05  # A frame must be this far past the cutoff to change state (as the chi1 cores).

    @classmethod
    def parse(cls, text: str) -> "DistanceSpec":
        m = _SPEC.match(text)
        if not m:
            raise ValueError(f"cannot read the distance {text!r}. Write it like A:88:OH-A:111:CG1@0.6 "
                             "(chain:residue:atom-chain:residue:atom@cutoff in nm).")
        ca, ra, na, cb, rb, nb, cut = m.groups()
        cutoff = float(cut)
        if not 0.1 < cutoff < 10:
            raise ValueError(f"the cutoff of {text!r} must be in nm, between 0.1 and 10.")
        return cls((ca, int(ra), na), (cb, int(rb), nb), cutoff)

    @property
    def label(self) -> str:
        def one(x):
            return (f"{x[0]}:" if x[0] else "") + f"{x[1]}:{x[2]}"
        return f"{one(self.a)}-{one(self.b)}@{self.cutoff_nm:g}"

    def residues(self) -> list[tuple]:
        return [(x[0], x[1]) for x in (self.a, self.b)]


def parse_distances(text: str | list | None) -> list[DistanceSpec]:
    """Several distances, separated by ';' (or a list of strings)."""
    if not text:
        return []
    parts = text if isinstance(text, list) else [p for p in text.split(";") if p.strip()]
    return [DistanceSpec.parse(p) for p in parts]


def find_atom(top: md.Topology, atom: tuple) -> int:
    chain, resseq, name = atom
    hits = [a.index for a in top.atoms
            if a.residue.resSeq == resseq and a.name == name and (chain is None or a.residue.chain.chain_id == chain)]
    if len(hits) != 1:
        where = (f"{chain}:" if chain else "") + f"{resseq}:{name}"
        raise ValueError(f"atom {where} {'is not in the structure' if not hits else 'is not unique'}. "
                         "Check the chain, the residue number and the atom name (for example CA, CB, OH, NZ).")
    return hits[0]


def distance_atoms(top: md.Topology, specs: list[DistanceSpec]) -> list[tuple[int, int]]:
    """Atom index pairs (in `top`) for each distance. Raises ValueError for an atom that is not there."""
    return [(find_atom(top, s.a), find_atom(top, s.b)) for s in specs]


def two_state(values: np.ndarray, cutoff: float, buffer: float) -> np.ndarray:
    """0 (near) below cutoff - buffer, 1 (far) above cutoff + buffer. Frames in between keep the last state;
    frames before the first clear state get -1."""
    v = np.asarray(values, dtype=float)
    core = np.full(v.shape, -1, dtype=int)
    core[v < cutoff - buffer] = 0
    core[v > cutoff + buffer] = 1
    states = core.copy()
    for t in range(1, len(states)):
        if states[t] == -1:
            states[t] = states[t - 1]
    return states


def diagnose_distances(trajs: list[md.Trajectory], specs: list[DistanceSpec], pairs: list[tuple[int, int]],
                       ns_per_replica: float, min_flips_per_replica: int = 4,
                       precision: float | None = None) -> list[ResidueReport]:
    """One report per distance, judged like the chi1 of a side chain (diagnose.judge).

    `pairs` are atom indices in the loaded trajectories.
    """
    reports = []
    for k, (spec, pair) in enumerate(zip(specs, pairs)):
        values = [md.compute_distances(t, [pair])[:, 0] for t in trajs]
        states = [two_state(v, spec.cutoff_nm, spec.buffer_nm) for v in values]
        occ = [occupancy(s, DISTANCE_STATES) for s in states]
        rep = ResidueReport(chain="", resseq=k, resname="", name=spec.label, quantity="distance",
                            transitions=[count_transitions(s) for s in states], occupancies=occ,
                            dominant=[max(o, key=o.get) for o in occ], ns_per_replica=ns_per_replica,
                            extra={"mean_nm_per_replica": [round(float(v.mean()), 3) for v in values],
                                   "cutoff_nm": spec.cutoff_nm})
        (rep.populations, rep.standard_error, rep.replicas_differ,
         rep.error_from_replicas) = populations_with_errors(states, DISTANCE_STATES)
        judge(rep, min_flips_per_replica, precision)
        reports.append(rep)
    return rank(reports)



CHI1_STATES = ["g+", "t", "g-"]   # The order of diagnose.STATE_NAMES.
_RESIDUE = re.compile(r"^\s*(?:([A-Za-z0-9]):)?(-?\d+)\s*$")


@dataclass
class EventSpec:
    """A change the user wants to see: item=state (see the module notes)."""
    text: str
    state: str
    residue: tuple | None = None          # (chain or None, number) for a chi1 item
    distance: DistanceSpec | None = None  # for a distance item

    @classmethod
    def parse(cls, text: str) -> "EventSpec":
        item, sep, state = text.strip().rpartition("=")
        state, item = state.strip(), item.strip()
        if not sep or not item:
            raise ValueError(f"cannot read the change {text!r}. Write it like A:111=t (side chain 111 enters t) or "
                             "A:88:OH-A:111:CG1@0.6=far (the distance becomes far).")
        if state in CHI1_STATES:
            m = _RESIDUE.match(item)
            if not m:
                raise ValueError(f"{item!r} is not a residue like A:111; chi1 states (g+, t, g-) need a residue.")
            return cls(text.strip(), state, residue=(m.group(1), int(m.group(2))))
        if state in DISTANCE_STATES:
            return cls(text.strip(), state, distance=DistanceSpec.parse(item))
        raise ValueError(f"unknown state {state!r} in {text!r}. Use g+, t or g- for a side chain, near or far "
                         "for a distance.")

    @property
    def label(self) -> str:
        if self.distance is not None:
            return f"{self.distance.label}={self.state}"
        chain, num = self.residue
        return (f"{chain}:" if chain else "") + f"{num}={self.state}"

    def residues(self) -> list[tuple]:
        return self.distance.residues() if self.distance is not None else [self.residue]

    @property
    def names(self) -> list[str]:
        return DISTANCE_STATES if self.distance is not None else CHI1_STATES


def parse_events(text: str | list | None) -> list[EventSpec]:
    parts = text if isinstance(text, list) else (text or "").split(";")
    specs = [EventSpec.parse(p) for p in parts if p and p.strip()]
    if not specs:
        raise ValueError("no change given. Write it like A:111=t or A:88:OH-A:111:CG1@0.6=far.")
    return specs


def first_entry(states: np.ndarray, target: int, hold_frames: int) -> int | None:
    """The frame at which the item has stayed `hold_frames` frames in state `target` after coming from another
    state. None if that did not happen. Frames between states (-1) keep the last state."""
    left, run, last = False, 0, -1
    for i, x in enumerate(np.asarray(states)):
        x = last if x < 0 else int(x)
        last = x
        if x < 0:
            continue
        if x == target:
            if left:
                run += 1
                if run >= hold_frames:
                    return i
        else:
            left, run = True, 0
    return None


def diagnose_events(series: list[list[np.ndarray]], specs: list[EventSpec], ns_per_replica: float, dt_ns: float,
                    hold_ps: float = 100.0, seen_in: int = 1) -> list[ResidueReport]:
    """One report per change. `series[k]` holds the per-frame states of item k, one array per replica.

    Verdict "ok" when the change was seen in at least `seen_in` replicas; "no_flip" when the item never changed
    state; otherwise "not_yet". Populations (time in each state) are given for information only.
    """
    hold = max(1, int(round(hold_ps / 1000.0 / dt_ns)))
    reports = []
    for k, (spec, states) in enumerate(zip(specs, series)):
        target = spec.names.index(spec.state)
        seen = [first_entry(s, target, hold) for s in states]
        seen_ns = [None if f is None else round((f + 1) * dt_ns, 3) for f in seen]
        occ = [occupancy(s, spec.names) for s in states]
        chain, num = (spec.residue if spec.residue else ("", k))
        rep = ResidueReport(chain=chain or "", resseq=num, resname="", name=spec.label, quantity="event",
                            transitions=[count_transitions(s) for s in states], occupancies=occ,
                            dominant=[max(o, key=o.get) for o in occ], ns_per_replica=ns_per_replica,
                            extra={"seen_ns": seen_ns, "target_state": spec.state, "hold_ps": hold_ps})
        (rep.populations, rep.standard_error, rep.replicas_differ,
         rep.error_from_replicas) = populations_with_errors(states, spec.names)
        rep.replicas_differ = rep.error_from_replicas = False   # Not a population goal: no replicas to add.
        n_seen = sum(t is not None for t in seen_ns)
        if n_seen >= seen_in:
            first = min(t for t in seen_ns if t is not None)
            rep.verdict = "ok"
            rep.reason = (f"Seen in {n_seen} of {len(states)} replicas, first at {first:g} ns "
                          f"(in {spec.state} for at least {hold_ps:g} ps after another state).")
        elif sum(rep.transitions) == 0:
            rep.verdict = "no_flip"
            rep.reason = f"Never changed state in {len(states)} replicas, so {spec.state} was not reached."
        else:
            rep.verdict = "not_yet"
            rep.reason = (f"Changes state ({rep.transitions} per replica), but has not stayed in {spec.state} for "
                          f"{hold_ps:g} ps yet" + (f" in {seen_in} replicas" if seen_in > 1 else "") + ".")
        reports.append(rep)
    return rank(reports)

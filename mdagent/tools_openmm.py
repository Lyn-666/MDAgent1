"""OpenMM tools and featurizers for the general decision loop (mdagent.core).

Featurizers turn the trajectories of a run into per-frame states (estimators.StateData), one per estimand:
- chi1 of each target side chain (states g+, t, g-; diagnose.assign_states), estimand "VAL111";
- distances (near/far; observables.two_state), estimand = the distance label;
- changes to see (event goal), estimand = the change label, for example "A:111=t".

Tools wrap the existing simulation code, so files on disk stay exactly as before:
- PlainMDTool ("plain_md", unbiased): rounds of run_segment for every replica; new replicas catch up.
  With sampling "adaptive" it is "adaptive_md" (seeded segments); its advance is still driven by
  agent.run_loop (phase 3 moves it here).
- MetadTool ("metadynamics", biased): run_replica on chi1, or on phi and psi for backbone goals (rates are not
  served: a bias changes the kinetics); observe rebuilds the bias with reweight.replay
  and returns per-frame weights (Tiwary-Parrinello), so the reweighted estimator can be used.

Every tool can also show an earlier moment of a finished run: PlainMDTool(parts=k, n_replicas=n) sees only
the first k rounds of the first n replicas; MetadTool(until_ns=d) only the first d ns. The bias rebuild is
causal (a sample's weight uses only earlier Gaussians), so a cut-off run gives the same weights as it did
at that time. The shadow replay (shadow.py) uses this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .estimators import StateData

POP, EVENT, RATES, LANDSCAPE = "state_populations", "event", "transition_rates", "free_energy_surface"


# ---------------------------------------------------------------- featurizers

def chi1_residues(run_dir: Path, target: list) -> dict[str, tuple]:
    """Target residues with a chi1 that can change state: {label: (chain, number)}. Pro and disulfide Cys are left out."""
    import mdtraj as md

    from .diagnose import _is_proline, fixed_chi1_residues

    t = md.load(str(Path(run_dir) / "protein.pdb"))
    idx, _ = md.compute_chi1(t)
    fixed = fixed_chi1_residues(Path(run_dir) / "protein.pdb")
    wanted = set(target)
    found = []
    for quad in idx:
        res = t.topology.atom(int(quad[1])).residue
        key = (res.chain.chain_id, res.resSeq)
        if key in wanted and key not in fixed and not _is_proline(res.name):
            found.append((f"{res.name}{res.resSeq}", key))
    names = [n for n, _ in found]
    return {(n if names.count(n) == 1 else f"{k[0]}:{n}"): k for n, k in found}


def event_series(run_dir: Path, state, n_parts: int, join: bool = True, n_traj: int | None = None) -> list:
    """Per-frame states of every change in state.events (a list per change, one array per trajectory).

    Shared with agent._diagnose_events, so the old and the new decisions see the same states.
    """
    import mdtraj as md

    from .diagnose import assign_states, load_run
    from .observables import distance_atoms, parse_events, two_state

    specs = parse_events(state.events)
    series: list = [None] * len(specs)
    chi_specs = [(k, sp) for k, sp in enumerate(specs) if sp.distance is None]
    if chi_specs:
        keys = sorted({sp.residue for _, sp in chi_specs}, key=lambda r: (r[0] or "", r[1]))
        top = md.load_topology(str(run_dir / "protein.pdb"))
        full = [(r.chain.chain_id, r.resSeq) for r in top.residues
                if any(r.resSeq == n and (c is None or r.chain.chain_id == c) for c, n in keys)]
        trajs = load_run(run_dir, full, n_parts, join=join)[:n_traj]
        idx, _ = md.compute_chi1(trajs[0][:1])
        where = {}
        for j, quad in enumerate(idx):
            res = trajs[0].topology.atom(int(quad[1])).residue
            where[(res.chain.chain_id, res.resSeq)] = j
        angles = [np.degrees(md.compute_chi1(t)[1]) for t in trajs]
        for k, sp in chi_specs:
            c, n = sp.residue
            match = [key for key in where if key[1] == n and (c is None or key[0] == c)]
            if not match:
                raise ValueError(f"residue {sp.label.split('=')[0]} has no chi1 angle (Gly, Ala) or is not in the run.")
            series[k] = [assign_states(a[:, where[match[0]]]) for a in angles]
    dist_specs = [(k, sp) for k, sp in enumerate(specs) if sp.distance is not None]
    if dist_specs:
        pairs = distance_atoms(md.load_topology(str(run_dir / "protein.pdb")), [sp.distance for _, sp in dist_specs])
        atoms = sorted({i for pair in pairs for i in pair})
        pos = {a: j for j, a in enumerate(atoms)}
        trajs = load_run(run_dir, state.target, n_parts, join=join, atoms=atoms)[:n_traj]
        for (k, sp), (i, j) in zip(dist_specs, pairs):
            values = [md.compute_distances(t, [(pos[i], pos[j])])[:, 0] for t in trajs]
            series[k] = [two_state(v, sp.distance.cutoff_nm, sp.distance.buffer_nm) for v in values]
    return series


def backbone_angles(run_dir: Path, state, n_parts: int, n_traj: int | None = None, join: bool = True):
    """phi and psi (degrees) of the target residues: ({label: (key, resname, [phi per traj], [psi per traj])},
    [(key, reason) left out]).

    Only the dihedral atoms are read, and each finished part (its .state.xml exists, so it will not change again)
    is read once: its angles are kept in backbone/cache/ and reused in later rounds. Long runs then cost one new
    part per round, not the whole trajectory again.
    """
    import json

    import mdtraj as md

    from .backbone import backbone_quads
    from .diagnose import _quiet_c_output
    from .simulate import part_path

    run_dir = Path(run_dir)
    top = md.load_topology(str(run_dir / "protein.pdb"))
    quads, skipped = backbone_quads(top, state.target)
    if not quads:
        return {}, skipped
    labels = list(quads)
    atoms = sorted({i for _, _, a, b in quads.values() for i in (*a, *b)})
    pos = {a: k for k, a in enumerate(atoms)}
    phi_q = [[pos[i] for i in quads[k][2]] for k in labels]
    psi_q = [[pos[i] for i in quads[k][3]] for k in labels]
    key = json.dumps([labels, [quads[k][2:] for k in labels]])
    cache = run_dir / "backbone" / "cache"
    replicas = sorted({int(p.name.split("_")[1]) for p in run_dir.glob("replica_*_part000.dcd")})
    if not replicas:
        raise FileNotFoundError(f"No trajectories in {run_dir}.")

    def part(i: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        stored = cache / f"replica_{i}_part{k:03d}.npz"
        if stored.exists():
            d = np.load(stored)
            if str(d["key"]) == key:
                return d["phi"], d["psi"]
        with _quiet_c_output():
            t = md.load_dcd(str(part_path(run_dir, i, k, ".dcd")), top=top, atom_indices=atoms)
        phi = np.degrees(md.compute_dihedrals(t, phi_q))   # frames x residues
        psi = np.degrees(md.compute_dihedrals(t, psi_q))
        if part_path(run_dir, i, k, ".state.xml").exists():   # Finished: it will not change again.
            cache.mkdir(parents=True, exist_ok=True)
            np.savez(stored, phi=phi, psi=psi, key=np.array(key))
        return phi, psi

    trajs = []
    for i in replicas:
        pieces = [part(i, k) for k in range(n_parts)]
        if join:
            trajs.append((np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces])))
        else:
            trajs.extend(pieces)
    trajs = trajs[:n_traj]
    out = {label: (quads[label][0], quads[label][1], [t[0][:, j] for t in trajs], [t[1][:, j] for t in trajs])
           for j, label in enumerate(labels)}
    return out, skipped


def backbone_estimands(label: str) -> dict[str, str]:
    """The three estimands of one residue's backbone, with their kinds."""
    return {f"{label} basins": POP, f"{label} transitions": RATES, f"{label} landscape": LANDSCAPE}


def estimand_kinds(run_dir: Path, state) -> dict[str, str]:
    """{estimand: kind} of a run, from its goal. Cheap: reads only protein.pdb."""
    q = state.quantity
    if q == "backbone_basins":
        import mdtraj as md

        from .backbone import backbone_quads
        quads, _ = backbone_quads(md.load_topology(str(Path(run_dir) / "protein.pdb")), state.target)
        return {e: k for label in quads for e, k in backbone_estimands(label).items()}
    if q == "event_seen":
        from .observables import parse_events
        return {sp.label: EVENT for sp in parse_events(state.events)}
    if q == "distance_states":
        from .observables import parse_distances
        return {sp.label: POP for sp in parse_distances(state.distances)}
    return {label: POP for label in chi1_residues(run_dir, state.target)}


def featurize(run_dir: Path, state, n_parts: int, n_replicas: int, join: bool = True) -> dict[str, StateData]:
    """StateData of every estimand of the run, from the first n_parts rounds of the first n_replicas replicas.

    join=False (adaptive sampling): every part of every replica is its own segment.
    """
    import mdtraj as md

    from .diagnose import STATE_NAMES, assign_states, load_run
    from .observables import CHI1_STATES, DISTANCE_STATES, distance_atoms, parse_distances, parse_events, two_state

    run_dir = Path(run_dir)
    n_traj = n_replicas if join else n_replicas * n_parts
    q = state.quantity
    if q == "backbone_basins":
        from .backbone import BackboneSettings, assign_basins, parse_basins

        from .estimators import AngleData

        settings = BackboneSettings.from_dict(state.backbone)
        basins = parse_basins(settings.basins)
        names = [b.name for b in basins]
        angles, _ = backbone_angles(run_dir, state, n_parts, n_traj, join)
        out = {}
        for label, (_, _, phis, psis) in angles.items():
            states = StateData([assign_basins(f, s, basins, settings.margin_deg) for f, s in zip(phis, psis)], names)
            e = backbone_estimands(label)
            k_pop, k_rates, k_land = list(e)
            out[k_pop] = out[k_rates] = states
            out[k_land] = AngleData(phis, psis, states, state.md.save_interval_ps / 1000.0)
        return out
    if q == "event_seen":
        specs = parse_events(state.events)
        series = event_series(run_dir, state, n_parts, join, n_traj)
        return {sp.label: StateData(s, list(sp.names)) for sp, s in zip(specs, series)}
    if q == "distance_states":
        specs = parse_distances(state.distances)
        pairs = distance_atoms(md.load_topology(str(run_dir / "protein.pdb")), specs)
        atoms = sorted({i for pair in pairs for i in pair})
        pos = {a: k for k, a in enumerate(atoms)}
        trajs = load_run(run_dir, state.target, n_parts, join=join, atoms=atoms)[:n_traj]
        out = {}
        for spec, (i, j) in zip(specs, pairs):
            values = [md.compute_distances(t, [(pos[i], pos[j])])[:, 0] for t in trajs]
            out[spec.label] = StateData([two_state(v, spec.cutoff_nm, spec.buffer_nm) for v in values],
                                        list(DISTANCE_STATES))
        return out
    residues = chi1_residues(run_dir, state.target)
    trajs = load_run(run_dir, state.target, n_parts, join=join)[:n_traj]
    idx, _ = md.compute_chi1(trajs[0][:1])
    column = {}
    for j, quad in enumerate(idx):
        res = trajs[0].topology.atom(int(quad[1])).residue
        column[(res.chain.chain_id, res.resSeq)] = j
    angles = [np.degrees(md.compute_chi1(t)[1]) for t in trajs]
    assert list(STATE_NAMES) == list(CHI1_STATES)
    return {label: StateData([assign_states(a[:, column[key]]) for a in angles], list(STATE_NAMES))
            for label, key in residues.items()}


# ---------------------------------------------------------------- tools

class PlainMDTool:
    cost_factor = 1.0

    def __init__(self, run_dir: Path, state, files=None, log: Callable[[str], None] = print,
                 parts: int | None = None, n_replicas: int | None = None):
        self.run_dir, self.state, self.files, self.log = Path(run_dir), state, files, log
        self.parts, self.n_replicas = parts, n_replicas     # None: everything on disk (a live tool).
        adaptive = state.sampling == "adaptive"
        self.name = "adaptive_md" if adaptive else "plain_md"
        self.regime = "seeded" if adaptive else "unbiased"
        self._kinds = None
        self._cache: dict = {}

    def _parts(self) -> int:
        return len(self.state.rounds_ns) if self.parts is None else self.parts

    def kinds(self) -> dict[str, str]:
        if self._kinds is None:
            self._kinds = estimand_kinds(self.run_dir, self.state)
        return self._kinds

    def supports(self, kind: str, estimand: str) -> bool:
        return self.kinds().get(estimand) == kind

    def length(self) -> float:
        return round(sum(self.state.rounds_ns[:self._parts()]), 9)

    def replicas(self) -> int:
        return self.state.n_replicas if self.n_replicas is None else self.n_replicas

    def observe(self, estimands: list) -> dict:
        key = (self._parts(), self.replicas())
        if key not in self._cache:
            self._cache = {key: featurize(self.run_dir, self.state, key[0], key[1], join=self.regime != "seeded")}
        return {e: self._cache[key][e] for e in estimands}

    def advance(self, n_replicas: int, until: float) -> None:
        """Run new replicas up to the current length, then one more round of `until - length` for all."""
        from .simulate import load_inputs, part_path, run_segment

        if self.parts is not None or self.n_replicas is not None:
            raise RuntimeError("a view of an earlier moment cannot run simulations")
        if self.regime == "seeded":
            raise NotImplementedError("adaptive sampling is still driven by agent.run_loop")
        s = self.state
        files = self.files = self.files or load_inputs(self.run_dir)
        for i in range(s.n_replicas, n_replicas):
            for p, ns in enumerate(s.rounds_ns):
                if not part_path(self.run_dir, i, p, ".state.xml").exists():
                    run_segment(files, self.run_dir, i, p, s.base_seed + 1000 * p + i, ns, s.md, self.log)
        s.n_replicas = max(s.n_replicas, n_replicas)
        s.save(self.run_dir)
        more = round(until - self.length(), 9)
        if more > 1e-9:
            part = len(s.rounds_ns)
            s.pending_round_ns = more
            s.save(self.run_dir)
            for i in range(s.n_replicas):
                if not part_path(self.run_dir, i, part, ".state.xml").exists():
                    run_segment(files, self.run_dir, i, part, s.base_seed + 1000 * part + i, more, s.md, self.log)
            s.rounds_ns.append(more)
            s.pending_round_ns = 0.0
            s.save(self.run_dir)
        self._cache = {}


class MetadTool:
    name, regime, cost_factor = "metadynamics", "biased", 1.0

    def __init__(self, run_dir: Path, state, files=None, log: Callable[[str], None] = print,
                 until_ns: float | None = None, n_replicas: int | None = None):
        from .enhanced import MetadSettings
        from .reweight import StopRule

        self.run_dir, self.state, self.files, self.log = Path(run_dir), state, files, log
        self.until_ns, self.n_replicas = until_ns, n_replicas
        es = state.enhanced
        self.settings = MetadSettings.from_dict(es.get("settings", {}))
        self.rule = StopRule(**{k: v for k, v in es.get("stop_rule", {}).items() if k in StopRule.__dataclass_fields__})
        self._candidates = None
        self._replay: dict = {}    # replica -> (number of samples, v, c), rebuilt when the file grows

    def chosen(self) -> dict[str, tuple]:
        """{label: (chain, number)} of the biased residues; empty before enhanced sampling started."""
        es = self.state.enhanced
        return {f"{es['resnames'][f'{c}:{n}']}{n}": (c, n) for c, n in es.get("residues", [])}

    @property
    def backbone(self) -> bool:
        return self.state.quantity == "backbone_basins"

    def _residue_of(self, kind: str, estimand: str) -> str | None:
        """The residue label an estimand belongs to, if this tool can serve that kind of estimand."""
        if self.backbone:
            for suffix, k in ((" basins", POP), (" landscape", LANDSCAPE)):   # Not " transitions": biased kinetics.
                if kind == k and estimand.endswith(suffix):
                    return estimand[: -len(suffix)]
            return None
        return estimand if kind == POP and self.state.quantity == "chi1_populations" else None

    def supports(self, kind: str, estimand: str) -> bool:
        label = self._residue_of(kind, estimand)
        if label is None or not self.state.enhanced.get("enabled", True):
            return False
        chosen = self.chosen()
        if chosen:
            return label in chosen
        if self._candidates is None:
            if self.backbone:
                import mdtraj as md

                from .backbone import backbone_quads
                quads, _ = backbone_quads(md.load_topology(str(self.run_dir / "protein.pdb")), self.state.target)
                self._candidates = set(quads)
            else:
                self._candidates = set(chi1_residues(self.run_dir, self.state.target))
        return label in self._candidates

    def _n(self) -> int:
        n = min(self.settings.n_replicas, self.state.n_replicas)
        return n if self.n_replicas is None else self.n_replicas

    def length(self) -> float:
        from .enhanced import ns_done

        if not self.chosen():
            return 0.0
        done = min(ns_done(self.run_dir, r) for r in range(self._n()))
        return done if self.until_ns is None else min(done, self.until_ns)

    def replicas(self) -> int:
        return self._n()

    def _rebuilt(self, r: int):
        from .enhanced import KB_KJ
        from .reweight import replay

        chi = np.load(self.run_dir / "enhanced" / f"replica_{r}_chi.npy")
        cached = self._replay.get(r)
        if cached is None or cached[0] != len(chi):
            v, c = replay(chi, self.settings, KB_KJ * self.state.md.temperature_k)
            self._replay[r] = cached = (len(chi), chi, v, c)
        return cached[1:]

    def observe(self, estimands: list) -> dict:
        from .diagnose import STATE_NAMES, assign_states
        from .enhanced import KB_KJ

        kt = KB_KJ * self.state.md.temperature_k
        labels = list(self.chosen())
        states = {k: [] for k in labels}
        weights = []
        if self.backbone:
            return self._observe_backbone(estimands, labels, kt)
        for r in range(self._n()):
            chi, v, c = self._rebuilt(r)
            frames = len(chi) if self.until_ns is None else min(len(chi), round(self.until_ns * 1000 / self.settings.pace_ps))
            if frames < 10:
                continue
            start = int(frames * self.rule.skip_fraction)
            logw = (v[start:frames] - c[start:frames]) / kt
            weights.append(np.exp(logw - logw.max()))
            for i, k in enumerate(labels):
                states[k].append(assign_states(np.degrees(chi[start:frames, i])))
        return {e: StateData(states[e], list(STATE_NAMES), weights) for e in estimands}

    def _observe_backbone(self, estimands: list, labels: list, kt: float) -> dict:
        """Basin states (weighted) and phi/psi (weighted) of each biased residue; columns are phi1, psi1, phi2, ..."""
        from .backbone import BackboneSettings, assign_basins, parse_basins
        from .estimators import AngleData

        bb = BackboneSettings.from_dict(self.state.backbone)
        basins = parse_basins(bb.basins)
        names = [b.name for b in basins]
        phi = {k: [] for k in labels}
        psi = {k: [] for k in labels}
        weights = []
        for r in range(self._n()):
            cols, v, c = self._rebuilt(r)
            frames = len(cols) if self.until_ns is None else min(len(cols), round(self.until_ns * 1000 / self.settings.pace_ps))
            if frames < 10:
                continue
            start = int(frames * self.rule.skip_fraction)
            logw = (v[start:frames] - c[start:frames]) / kt
            weights.append(np.exp(logw - logw.max()))
            deg = np.degrees(cols[start:frames])
            for i, k in enumerate(labels):
                phi[k].append(deg[:, 2 * i])
                psi[k].append(deg[:, 2 * i + 1])
        out = {}
        for e in estimands:
            label = e.rsplit(" ", 1)[0]
            states = StateData([assign_basins(f, s, basins, bb.margin_deg) for f, s in zip(phi[label], psi[label])],
                               names, weights)
            out[e] = (states if e.endswith(" basins")
                      else AngleData(phi[label], psi[label], states, self.settings.pace_ps / 1000.0, weights))
        return out

    def advance(self, n_replicas: int, until: float) -> None:
        """Run every enhanced sampling replica to `until` ns (new replicas start from the end of plain MD)."""
        from dataclasses import asdict

        from .enhanced import run_replica
        from .simulate import load_inputs, part_path

        if self.until_ns is not None or self.n_replicas is not None:
            raise RuntimeError("a view of an earlier moment cannot run simulations")
        es, s = self.state.enhanced, self.state
        if not es.get("residues"):
            raise ValueError("choose the residues to bias first (enhanced['residues'])")
        self.settings.n_replicas = n_replicas
        self.settings.length_ns = max(self.settings.length_ns, until)
        es["settings"] = asdict(self.settings)
        s.save(self.run_dir)
        files = self.files = self.files or load_inputs(self.run_dir)
        last = len(s.rounds_ns) - 1
        residues = [tuple(k) for k in es["residues"]]
        for r in range(n_replicas):
            start = part_path(self.run_dir, r % s.n_replicas, last, ".state.xml")
            run_replica(files, self.run_dir, r, start, residues, self.settings, s.md,
                        seed=s.base_seed + 50000 + r, log=self.log, until_ns=until,
                        kind="backbone" if self.backbone else "chi1")

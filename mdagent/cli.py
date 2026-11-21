"""Command line entry point.

Examples:
  # Start. The agent decides how long to run, up to --max-ns per replica.
  python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L --accept-problems

  # Continue a stopped run, or give a finished run a larger budget.
  python -m mdagent resume --run runs/181L --max-ns 40

  # Check protonation with PROPKA, use Amber ff19SB + OPC, 0.1 M KCl; build only.
  python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L_b --accept-problems \\
      --check-protonation --forcefield amber19 --ion-conc 0.1 --cation K+ --setup-only

  # Run enhanced sampling longer: 30 ns per replica instead of 20.
  python -m mdagent resume --run runs/181L --enhanced-ns 30

  # Show the latest ranked diagnosis of plain MD.
  python -m mdagent diagnose --run runs/181L
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _missing_packages() -> list[str]:
    """Required packages that this Python cannot import (usually: the mdagent conda environment is not active)."""
    return [name for name in ("mdtraj", "openmm", "pdbfixer", "pydantic") if importlib.util.find_spec(name) is None]


if _missing_packages():   # Say how to fix it, instead of a long traceback.
    env = os.environ.get("CONDA_DEFAULT_ENV")
    print(f"MDAgent cannot start: this Python ({sys.executable}) does not have {', '.join(_missing_packages())}.\n"
          + (f"The active conda environment is '{env}'. " if env else "")
          + "Activate the MDAgent environment first, then try again:\n"
            "    conda activate mdagent\n"
            "If that environment does not exist yet, install it as in README \"快速上手\" "
            "(conda env create -f environment.yml).\n"
            f"MDAgent 无法启动：这个 Python 里没有 {', '.join(_missing_packages())}。"
          + (f"当前的 conda 环境是“{env}”。" if env else "")
          + "请先运行 conda activate mdagent 再试；如果还没有这个环境，按 README“快速上手”安装。",
          file=sys.stderr)
    sys.exit(2)

from .target import Target, parse_residue_spec, resolve_target  # noqa: E402


def _target_from_args(args) -> Target:
    if args.around_ligand:
        return Target(ligand=args.around_ligand, cutoff_nm=args.cutoff_angstrom / 10.0)
    if not args.residues and getattr(args, "observe", None):
        # A run that waits for a change: the target is the residues named in the changes.
        from .observables import parse_events

        res = {(c, n) for s in parse_events(args.observe) for c, n in s.residues()}
        return Target(residues=tuple(sorted(res, key=lambda k: (k[0] or "", k[1]))))
    if not args.residues and getattr(args, "distances", None):
        # A distance run without a target region: the target is the residues of the distance atoms.
        from .observables import parse_distances

        res = {(c, n) for s in parse_distances(args.distances) for c, n in s.residues()}
        return Target(residues=tuple(sorted(res, key=lambda k: (k[0] or "", k[1]))))
    return Target(residues=parse_residue_spec(args.residues))


def _check_distances(args, structure: str | Path) -> list:
    """Parse --distances and find every atom in the structure. Raises ValueError with a clear message."""
    import mdtraj as md

    from .observables import distance_atoms, parse_distances

    specs = parse_distances(args.distances)
    distance_atoms(md.load_topology(str(structure)), specs)
    return specs


def _check_events(events, structure: str | Path) -> None:
    """Find every atom of the distance changes in the structure. Raises ValueError with a clear message."""
    import mdtraj as md

    from .observables import distance_atoms

    dists = [e.distance for e in events if e.distance is not None]
    if dists:
        distance_atoms(md.load_topology(str(structure)), dists)


def _print_summary(decisions, enhanced_labels, run_dir) -> None:
    from .outputs import describe

    flagged = [d.residue for d in decisions if d.action in ("enhanced_sampling", "stalled")]
    print("\nSummary")
    if not flagged:
        print("Plain MD sampled all target side chains well. No enhanced sampling was needed.")
    else:
        print("Plain MD could not sample: " + ", ".join(flagged) + ".")
        if enhanced_labels:
            print(f"Enhanced sampling finished for: {', '.join(enhanced_labels)}.")
            print(f"The order of residues in the .npy files: {', '.join(enhanced_labels)}.")
            pops = Path(run_dir) / "enhanced" / "populations.json"
            if pops.exists():
                import json

                from .reweight import format_table

                print(format_table(json.loads(pops.read_text())))
        else:
            print("Enhanced sampling did not run.")
    try:
        import json

        state = json.loads((Path(run_dir) / "run.json").read_text())
        for name, o in (("plain MD", state.get("notes", {}).get("plain_md_outcome")),
                        ("enhanced sampling", state.get("enhanced", {}).get("outcome"))):
            if o:
                print(f"Stop reason ({name}): {o['reason']} after {o['ns_per_replica']:g} ns per replica "
                      f"(budget {o['budget_ns']:g} ns).")
    except (OSError, ValueError, KeyError):
        pass
    print("\n".join(describe(run_dir)))


def _run_and_record(work, run_dir):
    """Run, and record stop reason "failed" (in run.json and the chat) if an error stops the run."""
    from .agent import record_failure

    try:
        result = work()
    except Exception as e:
        record_failure(Path(run_dir), e)
        raise
    _write_report(Path(run_dir))
    return result


def _write_report(run_dir: Path) -> None:
    """Step 6: the result report, and a message in the chat that says where it is."""
    from .messages import post
    from .report import write

    zh, en = write(run_dir)
    print(f"Result report: {en.resolve()} (Chinese: {zh.resolve()})")
    post(run_dir, f"完整的结果报告写好了：{zh.resolve()}（英文版 {en.name}）。里面有每个项目的比例和误差、是否达到要求、"
                  "停下的原因、采取过的动作、可信度和限制、文件位置，以及还剩的问题。",
         f"The full result report is ready: {en.resolve()} (Chinese: {zh.name}). It has every item's populations "
         "and errors, whether the goal was met, why it stopped, the actions taken, trust and limits, the files, "
         "and the open questions.", "done")


def _goal_from_args(args):
    from .agent import PERMISSIONS, Goal

    from .observables import QUANTITIES

    goal = Goal.for_mode(args.mode, description=args.goal, precision=args.precision,
                         quantity=QUANTITIES[getattr(args, "quantity", "chi1")],
                         max_replicas=getattr(args, "max_replicas", 10), extend_factor=getattr(args, "extend_factor", 2.0),
                         hold_ps=getattr(args, "hold_ps", 100.0), seen_in=getattr(args, "seen_in", 1),
                         rate_rel_error=getattr(args, "rate_rel_error", 0.25),
                         fes_error_kcal=getattr(args, "fes_error_kcal", 0.5))
    if args.allow is not None:
        allowed = [a.strip() for a in args.allow.split(",") if a.strip() and a.strip() != "none"]
        unknown = [a for a in allowed if a not in PERMISSIONS]
        if unknown:
            raise ValueError(f"unknown --allow value(s): {', '.join(unknown)}. Choose from {', '.join(PERMISSIONS)}.")
        goal.permissions = allowed
    if not 0 < goal.precision < 1:
        raise ValueError("--precision must be between 0 and 1.")
    return goal


def _host() -> str:
    from .outputs import host

    return host()


def _check_env(platform, forcefield_files=()) -> bool:
    from .envcheck import check_environment

    notes, problems = check_environment(platform, forcefield_files)
    for line in notes:
        print(line)
    for line in problems:
        print("error: " + line, file=sys.stderr)
    return not problems


def _chain_list(text):
    return [c.strip() for c in text.split(",") if c.strip()] if text else None


def _structure_check(args):
    from .check import check_structure

    print("Checking the structure ...")
    check = check_structure(args.pdb, chains=_chain_list(args.chains), keep_tags=args.keep_tags,
                            uniprot=args.uniprot, root=Path.cwd())
    print(check.report())
    termini = getattr(args, "termini", "auto")
    if termini != "auto":
        print(f"  note: chain ends: --termini {termini} replaces the suggestion above.")
    return check


def _build_from_pdb(args, keys, out, remove_residues=None, cap_ends=None):
    from .build import SetupOptions, build_system
    from .protonation import parse_overrides
    from .simulate import write_inputs

    setup = SetupOptions(
        forcefield=args.forcefield,
        ph=args.ph,
        ion_conc_molar=args.ion_conc,
        cation=args.cation,
        padding_nm=args.padding_nm,
        box_shape=args.box_shape,
        check_protonation=args.check_protonation,
        protonation={f"{c + ':' if c else ''}{n}": v
                     for (c, n), v in parse_overrides(args.protonation or "").items()},
        chains=_chain_list(args.chains),
        keep_tags=args.keep_tags,
        uniprot=args.uniprot,
        accept_problems=args.accept_problems,
        termini=args.termini,
    )
    setup.validate()
    print("Building the system" + (" (checking protonation states with PROPKA)" if setup.check_protonation else "")
          + " ...")
    built = build_system(args.pdb, setup, keys, remove_residues, cap_ends)
    report = built.report()
    print("\n".join("  " + line for line in report.splitlines()))
    write_inputs(built, out)
    (out / "setup_report.txt").write_text(report + "\n")

    return built, setup


# Options that only matter when MDAgent builds the system from a PDB file, with their defaults.
SETUP_DEFAULTS = {"forcefield": "charmm36m", "ph": 7.0, "ion_conc": 0.15, "cation": "Na+", "padding_nm": 1.0,
                  "box_shape": "cube", "check_protonation": False, "protonation": None, "chains": None,
                  "keep_tags": False, "uniprot": None, "accept_problems": False, "termini": "auto"}


def cmd_run(args) -> int:
    from dataclasses import asdict

    from .agent import LoopSettings, RunState, run_all
    from .build import SetupOptions, build_system
    from .protonation import parse_overrides
    from .simulate import MDSettings, SystemFiles, write_inputs

    from .build import FORCEFIELDS

    out = Path(args.out)
    print(f"Run folder: {out.resolve()} (on computer '{_host()}')")
    if (out / "run.json").exists():
        print(f"error: {out} already has a run. Use 'resume --run {out}' to continue it.",
              file=sys.stderr)
        return 2
    if args.observe:
        if args.quantity not in ("chi1", "event") or args.distances:
            print("error: --observe is a goal of its own; do not combine it with --quantity distance or --distances. "
                  "Write a distance change in --observe, for example \"A:88:OH-A:111:CG1@0.6=far\".", file=sys.stderr)
            return 2
        args.quantity = "event"
        print("Note: --observe given, so the run stops once the change is seen.")
        from .observables import parse_events
        try:
            events = parse_events(args.observe)
            if args.pdb:
                _check_events(events, args.pdb)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    elif args.quantity == "event":
        print("error: --quantity event needs --observe, for example \"A:111=t\".", file=sys.stderr)
        return 2
    if args.distances and args.quantity == "chi1":
        args.quantity = "distance"
        print("Note: --distances given, so the quantity is distance.")
    if args.quantity == "backbone":
        from .backbone import parse_basins
        try:
            parse_basins(args.basins)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.sampling == "adaptive":
            print("error: adaptive sampling does not support --quantity backbone yet.", file=sys.stderr)
            return 2
        if not (0 < args.rate_rel_error < 1) or args.fes_error_kcal <= 0:
            print("error: --rate-rel-error must be between 0 and 1, and --fes-error-kcal above 0.", file=sys.stderr)
            return 2
    if args.quantity in ("chi1", "backbone") and not (args.residues or args.around_ligand):
        print("error: give the target with --residues or --around-ligand.", file=sys.stderr)
        return 2
    if args.quantity == "distance":
        if not args.distances:
            print("error: --quantity distance needs --distances, for example \"A:88:OH-A:111:CG1@0.6\".",
                  file=sys.stderr)
            return 2
        if args.sampling == "adaptive":
            print("error: adaptive sampling does not support --quantity distance yet.", file=sys.stderr)
            return 2
        if args.pdb:
            _check_distances(args, args.pdb)  # Fail before building if an atom is not there.
    if args.enhanced_max_residues < 1:
        print("error: --enhanced-max-residues must be 1 or more.", file=sys.stderr)
        return 2
    amber = args.prmtop is not None
    if amber and not args.inpcrd:
        print("error: --prmtop needs --inpcrd.", file=sys.stderr)
        return 2
    if not _check_env(args.platform, () if amber else FORCEFIELDS[args.forcefield].files):
        return 2
    if args.sampling == "adaptive":
        from .adaptive import check_installed
        check_installed()

    if args.pdb_id:
        from .fetch import fetch_pdb

        path, new = fetch_pdb(args.pdb_id)
        print(("Downloaded" if new else "Using the downloaded file") + f" {path} (PDB ID {args.pdb_id.upper()}).")
        args.pdb = str(path)

    check = None
    if not amber:
        check = _structure_check(args)
        out.mkdir(parents=True, exist_ok=True)
        (out / "structure_check.txt").write_text(check.report() + "\n")
        if not check.clean and not args.accept_problems:
            print("\nThe run did not start. Deal with the problems above and run the command again (add the "
                  "options named under 'What to do'), or add --accept-problems to go on anyway.")
            return 4
        if not check.clean:
            print("Going on despite the problems above (--accept-problems).")

    target = _target_from_args(args)
    if amber:
        from .build import load_amber

        ignored = [f"--{k.replace('_', '-')}" for k, v in SETUP_DEFAULTS.items() if getattr(args, k) != v]
        if ignored:
            print("Note: these options change how a system is built, so they are ignored for a prepared "
                  "Amber system: " + ", ".join(ignored))
        print("Loading the prepared system ...")
        built = load_amber(args.prmtop, args.inpcrd)
        report = built.report()
        print("\n".join("  " + line for line in report.splitlines()))
        write_inputs(built, out)
        (out / "setup_report.txt").write_text(report + "\n")
        args.pdb = str(out / "solvated.pdb")
        setup = None
    keys = resolve_target(target, args.pdb)
    if check is not None:
        # Leave out target residues on removed chains or tags.
        removed = {(c, n) for c, n, _ in check.remove_residues}
        keys = [k for k in keys if k[0] in check.chains and tuple(k) not in removed]
        if not keys:
            print("error: no target residues are left on the kept chains.", file=sys.stderr)
            return 2
    print(f"Target: {target.describe()}")
    print(f"        {len(keys)} residues: " + ", ".join(f"{c}:{n}" for c, n in keys))

    if not target.ligand:
        asked = {(c, n) for c, n in target.residues}
        missing = sorted(n for c, n in asked if not any(k[1] == n and (c is None or k[0] == c) for k in keys))
        if missing:
            print("WARNING: these target residues have no coordinates in the PDB file and are "
                  "left out of the target: " + ", ".join(map(str, missing)))

    if check is not None:
        built, setup = _build_from_pdb(args, keys, out, check.remove_residues, check.cap_ends)
    specs = _check_distances(args, out / "protein.pdb") if args.quantity == "distance" else []
    from .observables import parse_events
    if args.quantity == "event":
        _check_events(parse_events(args.observe), out / "protein.pdb")

    loop = LoopSettings(
        first_ns=min(args.first_ns, args.max_ns),
        extend_ns=args.extend_ns,
        max_ns=args.max_ns,
        no_flip_patience_ns=args.no_flip_patience_ns,
    )
    state = RunState(
        pdb=str(args.pdb),
        target_description=target.describe(),
        target=keys,
        n_replicas=args.replicas,
        base_seed=args.seed,
        md=MDSettings(save_interval_ps=args.save_ps, platform=args.platform, equilibration_ps=args.equil_ps),
        loop=loop,
        setup=asdict(setup) if setup else {"amber": [str(args.prmtop), str(args.inpcrd)]},
        sampling=args.sampling,
        mode=args.mode,
        goal=asdict(_goal_from_args(args)),
        distances=[s.label for s in specs],
        backbone={"basins": args.basins, "margin_deg": args.basin_margin} if args.quantity == "backbone" else {},
        events=[e.label for e in parse_events(args.observe)] if args.quantity == "event" else [],
        adaptive={"policy": args.adaptive_policy, "n_clusters": args.adaptive_clusters,
                  "full_interval_ps": args.full_save_ps} if args.sampling == "adaptive" else {},
        enhanced={"enabled": not args.no_enhanced_sampling,
                  "settings": {"length_ns": args.enhanced_ns, "n_replicas": args.enhanced_replicas,
                               "max_residues": args.enhanced_max_residues}},
    )
    state.save(out)
    if args.setup_only:
        print(f"\nThe system is ready in {out}. Check the report above (also in setup_report.txt).")
        print(f"To start MD: python -m mdagent resume --run {out}")
        return 0
    goal = state.the_goal()
    if state.events:
        print(f"Goal: see the change {', '.join(state.events)} (the new state lasts at least {goal.hold_ps:g} ps, in "
              f"{goal.seen_in} replica(s)); stop once it is seen"
              + (f" ({goal.description})" if goal.description else "") + ".")
    else:
        what = (f"near/far populations of {', '.join(state.distances)}" if state.distances
                else "Ramachandran basin populations, transition statistics and phi/psi free-energy landscape of "
                     f"the target residues (rate error at most {goal.rate_rel_error:g}, landscape error at most "
                     f"{goal.fes_error_kcal:g} kcal/mol)" if state.quantity == "backbone_basins"
                else "chi1 populations of the target side chains")
        print(f"Goal: {what}, standard error at most {goal.precision:g}"
              + (f" ({goal.description})" if goal.description else "") + ".")
    print("The agent may by itself: " + (", ".join(goal.permissions) or "nothing (it only suggests)") + ".")
    print(f"Plan: {args.replicas} replicas. Round 1 is {loop.first_ns:g} ns per replica. "
          f"Later rounds add {loop.extend_ns:g} ns. Budget: {loop.max_ns:g} ns per replica.")
    if args.sampling == "adaptive":
        print(f"Adaptive sampling: from round 2, AdaptivePy ({args.adaptive_policy}) chooses where each "
              f"replica starts. Full-system frames are saved every {args.full_save_ps:g} ps.")
    if state.quantity == "backbone_basins":
        print("Enhanced sampling of backbone phi/psi is not available yet: if plain MD cannot sample a basin, the "
              "agent says so and suggests what to do.")
    elif state.enhanced["enabled"] and state.quantity == "chi1_populations":
        print(f"After plain MD, residues that still need it get enhanced sampling: "
              f"{args.enhanced_replicas} replicas x {args.enhanced_ns:g} ns, at most "
              f"{args.enhanced_max_residues} residues (the slowest ones).")
    files = SystemFiles(built.topology, built.positions, built.system)
    _print_summary(*_run_and_record(lambda: run_all(out, state, files), out), out)
    return 0


def cmd_resume(args) -> int:
    from .agent import RunState, run_all

    run = Path(args.run)
    state = RunState.load(run)
    changed = False
    if args.max_ns is not None:
        state.loop.max_ns = args.max_ns
        changed = True
    if args.replicas is not None:
        if args.replicas < state.n_replicas:
            print(f"error: the run has {state.n_replicas} replicas; --replicas can only add more.", file=sys.stderr)
            return 2
        changed = changed or args.replicas > state.n_replicas
        state.n_replicas = args.replicas
    if changed:
        # The user gave a new limit: decide again, and say again how plain MD ends.
        for key in ("plain_done", "said_extend", "said_no_gain", "said_add_replicas", "stalled"):
            state.notes.pop(key, None)
    if args.platform is not None:
        state.md.platform = args.platform
    if args.enhanced_sampling is not None:
        state.enhanced["enabled"] = args.enhanced_sampling
        state.enhanced["requested"] = bool(args.enhanced_sampling)  # The user asked for it (guided mode).
    if args.enhanced_ns is not None:
        state.enhanced.setdefault("settings", {})["length_ns"] = args.enhanced_ns
        state.enhanced.pop("outcome", None)  # The user asked for more: run it, then judge again.
    if not _check_env(state.md.platform):
        return 2
    state.save(run)
    print(f"Run folder: {run.resolve()} (on computer '{_host()}')")
    print(f"Resuming {run}: {state.ns_done:g} ns of plain MD per replica done. "
          f"Budget: {state.loop.max_ns:g} ns per replica.")
    _print_summary(*_run_and_record(lambda: run_all(run, state), run), run)
    return 0


def cmd_diagnose(args) -> int:
    from .agent import RunState, diagnose_run
    from .diagnose import format_report

    run = Path(args.run)
    state = RunState.load(run)
    if not state.rounds_ns:
        print("No round has finished yet.")
        return 0
    # The same diagnosis as the run itself: distances for a distance run, chi1 otherwise.
    print(format_report(diagnose_run(run, state)))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Write each line at once, also when the output goes to a log file (nohup ... > log).
    # Otherwise Python keeps lines in a buffer and the log falls behind.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    if args.command in ("run", "resume"):
        _record_how_the_run_ends()
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _record_how_the_run_ends() -> None:
    """Leave a trace in the log when a long run ends early.

    A crash in C code (for example in the GPU driver) and a stop signal (SIGTERM, also from the
    chat's stop button) write the Python stack to stderr, which goes to the log. The stop still
    happens at once. If a run ends with no trace, it was killed with SIGKILL (for example by the
    system when memory runs out) or the computer lost power.
    """
    import faulthandler
    import os
    import signal
    import time

    print(f"Process {os.getpid()} started at {time.strftime('%Y-%m-%d %H:%M:%S')}.")
    try:
        faulthandler.enable()
        faulthandler.register(signal.SIGTERM, chain=True)
    except (OSError, ValueError, AttributeError):
        pass  # stderr is not a real file (for example inside a test); nothing to write the stack to.


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mdagent")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Build the system and let the agent run MD until the target is sampled.")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", help="Protein PDB file. MDAgent builds the system (ligands are removed).")
    src.add_argument("--pdb-id", help="PDB ID, for example 3HTB. MDAgent downloads it to structures/<ID>.pdb "
                                     "(if it is not there yet) and uses that file like --pdb.")
    src.add_argument("--prmtop", help="A prepared, solvated Amber system (.parm7/.prmtop), for example from "
                                     "tleap with GAFF ligand parameters. Use it as it is; needs --inpcrd.")
    r.add_argument("--inpcrd", help="Coordinates and box for --prmtop (.rst7/.inpcrd).")
    _add_check_options(r)
    r.add_argument("--termini", default="auto", choices=["auto", "charged", "capped"],
                   help="Chain ends. auto (default): a neutral cap (ACE / NME) where the real protein continues "
                        "beyond the structure, charged NH3+ / COO- at real ends. charged: always charged. "
                        "capped: always capped.")
    r.add_argument("--accept-problems", action="store_true",
                   help="Build and run even if the structure check finds problems.")
    r.add_argument("--equil-ps", type=float, default=600.0,
                   help="Equilibration length: restrained heating, restrained NPT, restraint release, free NPT.")
    r.add_argument("--observe", default=None,
                   help="Stop once this change is seen, instead of measuring populations. item=state, separated by "
                        "';': \"A:111=t\" (side chain 111 enters t; states g+, t, g-) or "
                        "\"A:88:OH-A:111:CG1@0.6=far\" (the distance becomes far; near or far).")
    r.add_argument("--hold-ps", type=float, default=100.0,
                   help="--observe: the new state must last this long (ps) to count as seen (default 100).")
    r.add_argument("--seen-in", type=int, default=1,
                   help="--observe: in how many replicas the change must be seen (default 1).")
    r.add_argument("--quantity", default="chi1", choices=["chi1", "distance", "event", "backbone"],
                   help="What you want to know. chi1 (default): the rotamer states of the target side chains. "
                        "distance: how often atom distances are near or far (give them with --distances). "
                        "backbone: Ramachandran basin populations, transition statistics and the phi/psi "
                        "free-energy landscape of the target residues (give them with --residues).")
    r.add_argument("--basins", default="",
                   help="--quantity backbone: your own basins, 'name:phi_from,phi_to,psi_from,psi_to; ...' in degrees "
                        "(a name may repeat). Default: alpha_R, beta (with PPII), alpha_L, other.")
    r.add_argument("--basin-margin", type=float, default=10.0,
                   help="--quantity backbone: a frame is in a basin only this many degrees inside it (default 10).")
    r.add_argument("--rate-rel-error", type=float, default=0.25,
                   help="--quantity backbone: largest relative error of the leading transition rates (default 0.25).")
    r.add_argument("--fes-error-kcal", type=float, default=0.5,
                   help="--quantity backbone: largest error of the phi/psi free energy where it is at most 2 "
                        "kcal/mol (default 0.5).")
    r.add_argument("--distances", default=None,
                   help="For --quantity distance: atom pairs and cutoffs (nm), separated by ';', for example "
                        "\"A:88:OH-A:111:CG1@0.6;A:99:CA-A:133:CA@1.2\". The chain may be left out.")
    g = r.add_mutually_exclusive_group(required=False)
    g.add_argument("--residues", help="Target residues, for example 'A:99-120,A:133'.")
    g.add_argument("--around-ligand", help="Target = residues near this ligand (residue name in the PDB).")
    r.add_argument("--cutoff-angstrom", type=float, default=5.0)
    r.add_argument("--replicas", type=int, default=3)
    r.add_argument("--first-ns", type=float, default=1.0, help="Length of round 1 per replica (ns).")
    r.add_argument("--extend-ns", type=float, default=5.0, help="Length of each later round per replica (ns).")
    r.add_argument("--max-ns", type=float, default=20.0, help="Budget: total length per replica (ns).")
    r.add_argument("--no-flip-patience-ns", type=float, default=10.0,
                   help="Send a residue to enhanced sampling if it has not flipped after this (ns).")
    r.add_argument("--save-ps", type=float, default=5.0, help="Save a frame every this many ps.")
    r.add_argument("--forcefield", default="charmm36m", choices=["charmm36m", "amber14", "amber19"],
                   help="charmm36m: CHARMM36m + CHARMM TIP3P (default). amber14: ff14SB + TIP3P. "
                        "amber19: ff19SB + OPC.")
    r.add_argument("--ph", type=float, default=7.0)
    r.add_argument("--check-protonation", action="store_true",
                   help="Predict pKa values with PROPKA and use the predicted protonation states.")
    r.add_argument("--protonation", default=None,
                   help="Set protonation states by hand, for example 'A:31=HIP,A:70=ASH'. "
                        "Applied after --check-protonation.")
    r.add_argument("--ion-conc", type=float, default=0.15, help="Salt concentration in M (default 0.15).")
    r.add_argument("--cation", default="Na+", choices=["Na+", "K+"])
    r.add_argument("--padding-nm", type=float, default=1.0,
                   help="Minimum distance from the protein to each face of the water box (nm).")
    r.add_argument("--box-shape", default="cube", choices=["cube", "dodecahedron", "octahedron"],
                   help="A dodecahedron needs about 30%% fewer waters than a cube.")
    r.add_argument("--setup-only", action="store_true",
                   help="Build the system and stop. Start MD later with 'resume'.")
    r.add_argument("--platform", default=None, help="OpenMM platform: CUDA, OpenCL or CPU.")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--mode", default="auto", choices=["auto", "guided"],
                   help="auto (default): MDAgent chooses when to switch methods and when to stop. guided: run "
                        "only the method you chose; MDAgent suggests changes in the chat but does not switch.")
    r.add_argument("--goal", default="", help="What you want to know, in your own words (kept in run.json).")
    r.add_argument("--precision", type=float, default=0.05,
                   help="Largest allowed standard error of any chi1 state population (0 to 1). Default 0.05.")
    r.add_argument("--allow", default=None,
                   help="What the agent may do by itself, comma separated: extend, add_replicas, switch_method, "
                        "stop_early; or 'none'. Default: only stop_early in auto mode (time and replicas are "
                        "limits; the agent suggests the rest), none in guided mode.")
    r.add_argument("--max-replicas", type=int, default=10,
                   help="With permission add_replicas, the agent adds replicas up to this number (default 10).")
    r.add_argument("--extend-factor", type=float, default=2.0,
                   help="With permission extend, the agent may run up to this many times the first budget.")
    r.add_argument("--sampling", default="continue", choices=["continue", "adaptive"],
                   help="continue: each replica continues its own trajectory (default). adaptive: from "
                        "round 2, AdaptivePy chooses the starting frames (pip install adaptivepy-sampling).")
    r.add_argument("--adaptive-policy", default="least_counts", choices=["least_counts", "random"])
    r.add_argument("--adaptive-clusters", type=int, default=20)
    r.add_argument("--full-save-ps", type=float, default=10.0,
                   help="Adaptive mode: save all atoms (with water) every this many ps, for restarts.")
    r.add_argument("--enhanced-ns", type=float, default=20.0,
                   help="Length of each enhanced sampling replica (ns).")
    r.add_argument("--enhanced-replicas", type=int, default=2)
    r.add_argument("--enhanced-max-residues", type=int, default=10,
                   help="Bias at most this many residues at once in enhanced sampling (the slowest ones).")
    r.add_argument("--no-enhanced-sampling", action="store_true",
                   help="Stop after plain MD. Enhanced sampling can still run later with resume.")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    u = sub.add_parser("resume", help="Continue a run. Finishes a stopped round, then decides again.")
    u.add_argument("--run", required=True)
    u.add_argument("--max-ns", type=float, default=None, help="New budget per replica (ns).")
    u.add_argument("--replicas", type=int, default=None,
                   help="New number of plain MD replicas (more than now). New replicas first catch up.")
    u.add_argument("--platform", default=None)
    u.add_argument("--enhanced-ns", type=float, default=None,
                   help="New length of each enhanced sampling replica (ns). A larger value continues it.")
    es = u.add_mutually_exclusive_group()
    es.add_argument("--enhanced-sampling", dest="enhanced_sampling", action="store_true", default=None)
    es.add_argument("--no-enhanced-sampling", dest="enhanced_sampling", action="store_false")
    u.set_defaults(func=cmd_resume)

    d = sub.add_parser("diagnose", help="Print the ranked diagnosis of a run.")
    d.add_argument("--run", required=True)
    d.set_defaults(func=cmd_diagnose)

    k = sub.add_parser("check", help="Check a structure before a run: reference sequence, tags, chains, "
                                     "metals, gaps. Builds nothing.")
    ksrc = k.add_mutually_exclusive_group(required=True)
    ksrc.add_argument("--pdb")
    ksrc.add_argument("--pdb-id", help="Download to structures/<ID>.pdb first if needed.")
    _add_check_options(k)
    k.set_defaults(func=cmd_check)

    f = sub.add_parser("fetch", help="Download structures from the PDB to structures/<ID>.pdb.")
    f.add_argument("pdb_ids", nargs="+", help="One or more PDB IDs, for example 181L 3HTB.")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("info", help="Show the chains, residue ranges and ligands of a PDB or Amber file.")
    isrc = i.add_mutually_exclusive_group(required=True)
    isrc.add_argument("--pdb")
    isrc.add_argument("--prmtop", help="Amber topology (.parm7/.prmtop). Residue numbers start at 1.")
    i.set_defaults(func=cmd_info)

    rp = sub.add_parser("report", help="Write the result report of a run again (report.zh.md, report.en.md).")
    rp.add_argument("--run", required=True)
    rp.set_defaults(func=cmd_report)

    cp = sub.add_parser("compare", help="Do two runs agree within their errors? For example an early-stopped run "
                                        "and a longer reference run.")
    cp.add_argument("--run", required=True)
    cp.add_argument("--reference", required=True)
    cp.set_defaults(func=cmd_compare)

    sh = sub.add_parser("shadow", help="What would the new decision core have done? Replays every decision point "
                                       "of a run from its files (no new simulation) and compares with what the run did.")
    sh.add_argument("--run", required=True, nargs="+", help="One or more run folders.")
    sh.add_argument("--live", action="store_true", help="Compare the live shadow records (decisions.core.jsonl) "
                                                        "instead of replaying.")
    sh.set_defaults(func=cmd_shadow)

    st = sub.add_parser("start", help="Start here: choose the language, the language model (Claude or ChatGPT) "
                                      "and where to talk (web page or terminal).")
    st.add_argument("--backend", default=None, choices=["claude", "codex", "api"],
                    help="Language model. Not given: MDAgent asks.")
    st.add_argument("--ui", default=None, choices=["web", "terminal"], help="Not given: MDAgent asks.")
    st.add_argument("--lang", default=None, choices=["zh", "en"], help="Not given: MDAgent asks.")
    st.add_argument("--port", type=int, default=8765)
    st.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    st.set_defaults(func=cmd_start)

    c = sub.add_parser("chat", help="Talk with MDAgent: a local web page, or the terminal with --terminal.")
    c.add_argument("--backend", default="auto", choices=["auto", "claude", "codex", "api"],
                   help="Language model: claude (Claude Code login), codex (Codex login), api (ANTHROPIC_API_KEY). "
                        "auto picks the first one installed.")
    c.add_argument("--terminal", action="store_true", help="Talk in this terminal instead of a web page.")
    c.add_argument("--lang", default="zh", choices=["zh", "en"], help="Language of the terminal mode.")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    c.set_defaults(func=cmd_chat)

    a = sub.add_parser("ask", help="Describe the run in one sentence. The language model turns it into settings; "
                                   "you confirm before anything runs.")
    a.add_argument("request", help='For example: "use CHARMM, look at the pocket around benzene, 0.1 M KCl".')
    a.add_argument("--pdb", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--yes", action="store_true", help="Start without asking for confirmation.")
    a.add_argument("--backend", default="auto", choices=["auto", "claude", "codex", "api"],
                   help="Language model, as for chat. auto picks the first one installed.")
    a.set_defaults(func=cmd_ask)
    return p


def cmd_ask(args) -> int:
    from .llm import ask

    return ask(args.request, args.pdb, args.out, assume_yes=args.yes, backend=args.backend)


def cmd_info(args) -> int:
    from .target import pdb_summary

    print(pdb_summary(args.pdb or args.prmtop))
    return 0


def cmd_chat(args) -> int:
    from . import backends
    from .chat import Assistant, Runs, serve
    from .terminal import run_terminal

    if args.terminal:
        root = Path.cwd()
        assistant = Assistant(root=root, backend=backends.choose(args.backend), runs=Runs(root), ui="terminal")
        return run_terminal(assistant, args.lang)
    serve(Path.cwd(), args.backend, args.port, open_browser=not args.no_browser)
    return 0


def cmd_report(args) -> int:
    from .report import write

    for path in write(Path(args.run)):
        print(f"Written: {path.resolve()}")
    return 0


def cmd_compare(args) -> int:
    from .report import compare

    table, agree = compare(Path(args.run), Path(args.reference))
    print(table)
    return 0 if agree else 1


def cmd_shadow(args) -> int:
    from . import shadow

    status = 0
    for run in args.run:
        run_dir = Path(run)
        print(f"== {run_dir.resolve()}")
        try:
            if args.live:
                records = shadow.load(run_dir / shadow.LIVE_FILE)
                if not records:
                    print(f"No live shadow records in {run_dir / shadow.LIVE_FILE} (runs started before this "
                          "version have none; use the replay without --live).")
                    continue
            else:
                records = shadow.replay(run_dir, log=lambda line: None)
            table, _ = shadow.compare(run_dir, records)
            print(table)
            if not args.live:
                print(f"Core decisions with their evidence: {(run_dir / shadow.REPLAY_FILE).resolve()}")
        except Exception as e:
            print(f"Could not compare {run_dir}: {type(e).__name__}: {e}")
            status = 1
        print()
    return status


def cmd_start(args) -> int:
    from .terminal import start

    return start(Path.cwd(), args.backend, args.ui, args.lang, args.port, open_browser=not args.no_browser)


def cmd_fetch(args) -> int:
    from .fetch import fetch_pdb
    from .target import pdb_summary

    for pdb_id in args.pdb_ids:
        path, new = fetch_pdb(pdb_id)
        print(("Downloaded " if new else "Already there: ") + str(path))
        print("  " + pdb_summary(path).replace("\n", "\n  "))
    return 0


def _add_check_options(p) -> None:
    p.add_argument("--chains", help="Keep only these protein chains, for example 'A' or 'A,B'.")
    p.add_argument("--keep-tags", action="store_true", help="Keep tag residues at the chain ends.")
    p.add_argument("--uniprot", help="UniProt accession of the reference sequence: for files without one, or to "
                                     "replace the file's (for example an entry that is no longer active).")


def cmd_check(args) -> int:
    if args.pdb_id:
        from .fetch import fetch_pdb

        path, new = fetch_pdb(args.pdb_id)
        print(("Downloaded " if new else "Using ") + str(path))
        args.pdb = str(path)
    return 0 if _structure_check(args).clean else 4

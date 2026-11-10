"""Command line entry point.

Examples:
  # Start. The agent decides how long to run, up to --max-ns per replica.
  python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L

  # Continue a stopped run, or give a finished run a larger budget.
  python -m mdagent resume --run runs/181L --max-ns 40

  # Show the latest ranked diagnosis.
  python -m mdagent diagnose --run runs/181L
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .target import Target, parse_residue_spec, resolve_target


def _target_from_args(args) -> Target:
    if args.around_ligand:
        return Target(ligand=args.around_ligand, cutoff_nm=args.cutoff_angstrom / 10.0)
    return Target(residues=parse_residue_spec(args.residues))


def _print_decisions(decisions) -> None:
    enhanced = [d.residue for d in decisions if d.action == "enhanced_sampling"]
    print("\nThe agent stopped plain MD.")
    if enhanced:
        print("Residues that need enhanced sampling, slowest first: " + ", ".join(enhanced))
    else:
        print("All target side chains are sampled well.")


def cmd_run(args) -> int:
    from .agent import LoopSettings, RunState, run_loop
    from .build import build_system
    from .simulate import MDSettings, SystemFiles, write_inputs

    out = Path(args.out)
    if (out / "run.json").exists():
        print(f"error: {out} already has a run. Use 'resume --run {out}' to continue it.",
              file=sys.stderr)
        return 2

    target = _target_from_args(args)
    keys = resolve_target(target, args.pdb)
    print(f"Target: {target.describe()}")
    print(f"        {len(keys)} residues: " + ", ".join(f"{c}:{n}" for c, n in keys))

    print("Building the system ...")
    built = build_system(args.pdb, ph=args.ph, padding_nm=args.padding_nm)
    for note in built.notes:
        print("  " + note)
    write_inputs(built, out)

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
        md=MDSettings(save_interval_ps=args.save_ps, platform=args.platform),
        loop=loop,
    )
    state.save(out)
    print(f"Plan: {args.replicas} replicas. Round 1 is {loop.first_ns:g} ns per replica. "
          f"Later rounds add {loop.extend_ns:g} ns. Budget: {loop.max_ns:g} ns per replica.")
    files = SystemFiles(built.topology, built.positions, built.system)
    _print_decisions(run_loop(out, state, files))
    return 0


def cmd_resume(args) -> int:
    from .agent import RunState, run_loop

    run = Path(args.run)
    state = RunState.load(run)
    if args.max_ns is not None:
        state.loop.max_ns = args.max_ns
    if args.platform is not None:
        state.md.platform = args.platform
    state.save(run)
    print(f"Resuming {run}: {state.ns_done:g} ns per replica done. "
          f"Budget: {state.loop.max_ns:g} ns per replica.")
    _print_decisions(run_loop(run, state))
    return 0


def cmd_diagnose(args) -> int:
    from .agent import RunState
    from .diagnose import diagnose_trajectories, format_report, load_run

    run = Path(args.run)
    state = RunState.load(run)
    if not state.rounds_ns:
        print("No round has finished yet.")
        return 0
    reports = diagnose_trajectories(
        load_run(run, state.target, len(state.rounds_ns)), state.target, state.ns_done,
        min_flips_per_replica=state.loop.min_flips_per_replica)
    print(format_report(reports))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mdagent")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Build the system and let the agent run MD until the target is sampled.")
    r.add_argument("--pdb", required=True)
    g = r.add_mutually_exclusive_group(required=True)
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
    r.add_argument("--ph", type=float, default=7.0)
    r.add_argument("--padding-nm", type=float, default=1.0)
    r.add_argument("--platform", default=None, help="OpenMM platform: CUDA, OpenCL or CPU.")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    u = sub.add_parser("resume", help="Continue a run. Finishes a stopped round, then decides again.")
    u.add_argument("--run", required=True)
    u.add_argument("--max-ns", type=float, default=None, help="New budget per replica (ns).")
    u.add_argument("--platform", default=None)
    u.set_defaults(func=cmd_resume)

    d = sub.add_parser("diagnose", help="Print the ranked diagnosis of a run.")
    d.add_argument("--run", required=True)
    d.set_defaults(func=cmd_diagnose)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

"""Command line entry point.

Examples:
  python -m binding_lang run --pdb examples/181L.pdb --around-ligand BNZ \\
      --replicas 3 --ns 1 --out runs/181L
  python -m binding_lang diagnose --run runs/181L
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .target import Target, parse_residue_spec, resolve_target


def _target_from_args(args) -> Target:
    if args.around_ligand:
        return Target(ligand=args.around_ligand, cutoff_nm=args.cutoff_angstrom / 10.0)
    return Target(residues=parse_residue_spec(args.residues))


def cmd_run(args) -> int:
    from .build import build_system
    from .diagnose import diagnose_trajectories, format_report, load_run
    from .simulate import MDSettings, run_all

    out = Path(args.out)
    target = _target_from_args(args)
    keys = resolve_target(target, args.pdb)
    print(f"Target: {target.describe()}")
    print(f"        {len(keys)} residues: " + ", ".join(f"{c}:{n}" for c, n in keys))

    print("Building the system ...")
    built = build_system(args.pdb, ph=args.ph, padding_nm=args.padding_nm)
    for note in built.notes:
        print("  " + note)

    settings = MDSettings(
        production_ns=args.ns,
        equilibration_ps=args.equil_ps,
        save_interval_ps=args.save_ps,
        platform=args.platform,
    )
    print(f"Running {args.replicas} replicas of {args.ns:g} ns ...")
    run_all(built, out, args.replicas, settings, base_seed=args.seed)
    (out / "target.json").write_text(json.dumps(
        {"description": target.describe(), "residues": keys}, indent=2))

    print("\nDiagnosis of the target side chains:\n")
    reports = diagnose_trajectories(load_run(out), keys)
    text = format_report(reports, args.ns)
    print(text)
    (out / "diagnosis.txt").write_text(text + "\n")
    return 0


def cmd_diagnose(args) -> int:
    from .diagnose import diagnose_trajectories, format_report, load_run

    run = Path(args.run)
    keys = [tuple(k) for k in json.loads((run / "target.json").read_text())["residues"]]
    ns = json.loads((run / "run.json").read_text())["settings"]["production_ns"]
    reports = diagnose_trajectories(load_run(run), keys, min_flips_per_replica=args.min_flips)
    print(format_report(reports, ns))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="binding_lang")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Build the system, run MD, and diagnose the target side chains.")
    r.add_argument("--pdb", required=True)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--residues", help="Target residues, for example 'A:99-120,A:133'.")
    g.add_argument("--around-ligand", help="Target = residues near this ligand (residue name in the PDB).")
    r.add_argument("--cutoff-angstrom", type=float, default=5.0)
    r.add_argument("--replicas", type=int, default=3)
    r.add_argument("--ns", type=float, default=1.0, help="Production length per replica, in ns.")
    r.add_argument("--equil-ps", type=float, default=100.0)
    r.add_argument("--save-ps", type=float, default=2.0)
    r.add_argument("--ph", type=float, default=7.0)
    r.add_argument("--padding-nm", type=float, default=1.0)
    r.add_argument("--platform", default=None, help="OpenMM platform: CUDA, OpenCL or CPU.")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("diagnose", help="Diagnose an existing run directory again.")
    d.add_argument("--run", required=True)
    d.add_argument("--min-flips", type=int, default=4)
    d.set_defaults(func=cmd_diagnose)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

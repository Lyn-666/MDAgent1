"""Command line entry point.

Examples:
  # Start. The agent decides how long to run, up to --max-ns per replica.
  python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L

  # Continue a stopped run, or give a finished run a larger budget.
  python -m mdagent resume --run runs/181L --max-ns 40

  # Check protonation with PROPKA, use Amber ff19SB + OPC, 0.1 M KCl; build only.
  python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L_b \\
      --check-protonation --forcefield amber19 --ion-conc 0.1 --cation K+ --setup-only

  # Run enhanced sampling longer: 30 ns per replica instead of 20.
  python -m mdagent resume --run runs/181L --enhanced-ns 30

  # Show the latest ranked diagnosis of plain MD.
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


def _print_summary(decisions, enhanced_labels, run_dir) -> None:
    flagged = [d.residue for d in decisions if d.action == "enhanced_sampling"]
    print("\nSummary")
    if not flagged:
        print("Plain MD sampled all target side chains well. No enhanced sampling was needed.")
        return
    print("Plain MD could not sample: " + ", ".join(flagged) + ".")
    if not enhanced_labels:
        print("Enhanced sampling did not run.")
        return
    d = Path(run_dir) / "enhanced"
    print(f"Enhanced sampling finished for: {', '.join(enhanced_labels)}.")
    print(f"Output files in {d}:")
    print("  replica_<i>_chunk<k>.dcd   protein trajectory of each 1 ns chunk (topology: protein.pdb)")
    print("  replica_<i>_chi.npy        chi1 of each biased residue (radians), one row per pace")
    print("  replica_<i>_bias.npy       final bias of each biased residue on the chi1 grid (kJ/mol)")
    print(f"  The order of residues in the .npy files: {', '.join(enhanced_labels)}.")


def _check_env(platform, forcefield_files=()) -> bool:
    from .envcheck import check_environment

    notes, problems = check_environment(platform, forcefield_files)
    for line in notes:
        print(line)
    for line in problems:
        print("error: " + line, file=sys.stderr)
    return not problems


def cmd_run(args) -> int:
    from dataclasses import asdict

    from .agent import LoopSettings, RunState, run_all
    from .build import SetupOptions, build_system
    from .protonation import parse_overrides
    from .simulate import MDSettings, SystemFiles, write_inputs

    from .build import FORCEFIELDS

    out = Path(args.out)
    if (out / "run.json").exists():
        print(f"error: {out} already has a run. Use 'resume --run {out}' to continue it.",
              file=sys.stderr)
        return 2
    if not _check_env(args.platform, FORCEFIELDS[args.forcefield].files):
        return 2

    target = _target_from_args(args)
    keys = resolve_target(target, args.pdb)
    print(f"Target: {target.describe()}")
    print(f"        {len(keys)} residues: " + ", ".join(f"{c}:{n}" for c, n in keys))

    if not target.ligand:
        asked = {(c, n) for c, n in target.residues}
        missing = sorted(n for c, n in asked if not any(k[1] == n and (c is None or k[0] == c) for k in keys))
        if missing:
            print("WARNING: these target residues have no coordinates in the PDB file and are "
                  "left out of the target: " + ", ".join(map(str, missing)))

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
    )
    setup.validate()
    print("Building the system" + (" (checking protonation states with PROPKA)" if setup.check_protonation else "")
          + " ...")
    built = build_system(args.pdb, setup, keys)
    report = built.report()
    print("\n".join("  " + line for line in report.splitlines()))
    write_inputs(built, out)
    (out / "setup_report.txt").write_text(report + "\n")

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
        setup=asdict(setup),
        enhanced={"enabled": not args.no_enhanced_sampling,
                  "settings": {"length_ns": args.enhanced_ns, "n_replicas": args.enhanced_replicas}},
    )
    state.save(out)
    if args.setup_only:
        print(f"\nThe system is ready in {out}. Check the report above (also in setup_report.txt).")
        print(f"To start MD: python -m mdagent resume --run {out}")
        return 0
    print(f"Plan: {args.replicas} replicas. Round 1 is {loop.first_ns:g} ns per replica. "
          f"Later rounds add {loop.extend_ns:g} ns. Budget: {loop.max_ns:g} ns per replica.")
    if state.enhanced["enabled"]:
        print(f"After plain MD, residues that still need it get enhanced sampling: "
              f"{args.enhanced_replicas} replicas x {args.enhanced_ns:g} ns.")
    files = SystemFiles(built.topology, built.positions, built.system)
    _print_summary(*run_all(out, state, files), out)
    return 0


def cmd_resume(args) -> int:
    from .agent import RunState, run_all

    run = Path(args.run)
    state = RunState.load(run)
    if args.max_ns is not None:
        state.loop.max_ns = args.max_ns
    if args.platform is not None:
        state.md.platform = args.platform
    if args.enhanced_sampling is not None:
        state.enhanced["enabled"] = args.enhanced_sampling
    if args.enhanced_ns is not None:
        state.enhanced.setdefault("settings", {})["length_ns"] = args.enhanced_ns
    if not _check_env(state.md.platform):
        return 2
    state.save(run)
    print(f"Resuming {run}: {state.ns_done:g} ns of plain MD per replica done. "
          f"Budget: {state.loop.max_ns:g} ns per replica.")
    _print_summary(*run_all(run, state), run)
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
    r.add_argument("--enhanced-ns", type=float, default=20.0,
                   help="Length of each enhanced sampling replica (ns).")
    r.add_argument("--enhanced-replicas", type=int, default=2)
    r.add_argument("--no-enhanced-sampling", action="store_true",
                   help="Stop after plain MD. Enhanced sampling can still run later with resume.")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    u = sub.add_parser("resume", help="Continue a run. Finishes a stopped round, then decides again.")
    u.add_argument("--run", required=True)
    u.add_argument("--max-ns", type=float, default=None, help="New budget per replica (ns).")
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

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

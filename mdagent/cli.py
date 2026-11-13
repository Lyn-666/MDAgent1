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


def _chain_list(text):
    return [c.strip() for c in text.split(",") if c.strip()] if text else None


def _structure_check(args):
    from .check import check_structure

    print("Checking the structure ...")
    check = check_structure(args.pdb, chains=_chain_list(args.chains), keep_tags=args.keep_tags,
                            uniprot=args.uniprot, root=Path.cwd())
    print(check.report())
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
    if (out / "run.json").exists():
        print(f"error: {out} already has a run. Use 'resume --run {out}' to continue it.",
              file=sys.stderr)
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
        adaptive={"policy": args.adaptive_policy, "n_clusters": args.adaptive_clusters,
                  "full_interval_ps": args.full_save_ps} if args.sampling == "adaptive" else {},
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
    if args.sampling == "adaptive":
        print(f"Adaptive sampling: from round 2, AdaptivePy ({args.adaptive_policy}) chooses where each "
              f"replica starts. Full-system frames are saved every {args.full_save_ps:g} ps.")
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
    # Write each line at once, also when the output goes to a log file (nohup ... > log).
    # Otherwise Python keeps lines in a buffer and the log falls behind.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


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

    c = sub.add_parser("chat", help="Open the MDAgent chat window (a local web page).")
    c.add_argument("--backend", default="auto", choices=["auto", "claude", "codex", "api"],
                   help="Language model: claude (Claude Code login), codex (Codex login), api (ANTHROPIC_API_KEY). "
                        "auto picks the first one installed.")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    c.set_defaults(func=cmd_chat)

    a = sub.add_parser("ask", help="Describe the run in plain language. Claude turns it into settings; "
                                   "you confirm before anything runs.")
    a.add_argument("request", help='For example: "use CHARMM, look at the pocket around benzene, 0.1 M KCl".')
    a.add_argument("--pdb", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--yes", action="store_true", help="Start without asking for confirmation.")
    a.set_defaults(func=cmd_ask)
    return p


def cmd_ask(args) -> int:
    from .llm import ask

    return ask(args.request, args.pdb, args.out, assume_yes=args.yes)


def cmd_info(args) -> int:
    from .target import pdb_summary

    print(pdb_summary(args.pdb or args.prmtop))
    return 0


def cmd_chat(args) -> int:
    from .chat import serve

    serve(Path.cwd(), args.backend, args.port, open_browser=not args.no_browser)
    return 0


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
    p.add_argument("--uniprot", help="UniProt accession of the reference sequence, for files without one.")


def cmd_check(args) -> int:
    if args.pdb_id:
        from .fetch import fetch_pdb

        path, new = fetch_pdb(args.pdb_id)
        print(("Downloaded " if new else "Using ") + str(path))
        args.pdb = str(path)
    return 0 if _structure_check(args).clean else 4

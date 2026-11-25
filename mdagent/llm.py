"""Plain-language requests: a language model fills in the run settings, the user confirms.

The language model (Claude Code, Codex or the Anthropic API, see backends.py) only fills in RunPlan, a fixed form whose fields are the options of
`mdagent run`. A field left empty keeps its default. The form is turned into
a normal `mdagent run` command line and checked by the same parser as the
command line. The user sees the settings and the command, and confirms
before anything runs. If the request is unclear, Claude asks questions
instead, and nothing runs.

The `ask` command works with the same backends as the chat window: a Claude
Code or Codex login, or an API key.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .target import pdb_summary  # noqa: F401  (used by ask, and re-exported)

MODEL = "claude-opus-5-5"


class RunPlan(BaseModel):
    """Settings for one `mdagent run`. Leave a field null to keep its default."""

    target_kind: Literal["ligand", "residues", "distance", "backbone", "change", "stability"] = Field(
        description="'ligand': the target is all protein residues near a ligand in the PDB file. "
                    "'residues': the target is a list of residues. Both study side-chain conformations (chi1). "
                    "'distance': the user wants to know about distances between atoms (a pocket opening, a "
                    "contact, a gate); give them in `distances`. "
                    "'backbone': the user wants backbone conformations: Ramachandran (phi/psi) basin populations, "
                    "transitions between them, a phi/psi free-energy landscape, or the type of a beta turn; give "
                    "them in `backbone`. "
                    "'change': the user wants to stop once a given change is seen (a side chain reaches a state, "
                    "a pocket opens); give it in `observe`. "
                    "'stability': the user wants to know whether the structure stays stable (backbone RMSD from the "
                    "start, radius of gyration, steady state, unfolding, an RMSD-Rg free-energy surface, "
                    "conformational clusters); the whole protein, or a region in `residues`. "
                    "'backbone': the user wants backbone results (Ramachandran / phi-psi basin populations, transitions "
                    "between basins, a phi/psi free-energy landscape, the type of a beta turn); give the items in "
                    "`backbone`, or only the residues in `residues` (each becomes a rama item).")
    backbone: Optional[str] = Field(
        None, description="Only for target_kind 'backbone', separated by ';': 'rama:A:73' for the phi/psi basins of "
                          "residue 73 ('rama:A:70-74' for a range), 'turn:A:52' for the beta turn whose residue i+1 "
                          "is 52 (type I or II). Use residue numbers from the file summary; ask if they are unclear.")
    basins: Optional[str] = Field(
        None, description="Only for 'backbone', and only if the user defines their own Ramachandran regions: "
                          "'name:phi_from,phi_to,psi_from,psi_to; ...' in degrees. Null: the default regions.")
    rate_rel_error: Optional[float] = Field(
        None, description="Only for 'backbone': largest relative error of the leading transition rates, if the "
                          "user states one (for example 'rates within 20%' -> 0.2). Default 0.25.")
    fes_error_kcal: Optional[float] = Field(
        None, description="Only for 'backbone': largest error of the free-energy landscape in kcal/mol, if the user "
                          "states one. Default 0.5.")
    allocation: Optional[Literal["rounds", "projection"]] = Field(
        None, description="Leave null (default): MDAgent plans how many replicas, how long and when to stop "
                          "(projection) unless the user names the method, asks for adaptive sampling or watches for a "
                          "change (then rounds). Set 'rounds' only if the user asks for a fixed plan, for example "
                          "'run exactly 3 replicas of 10 ns'.")
    budget_ns: Optional[float] = Field(
        None, description="Only if the user gives a total budget: production ns of all replicas together.")
    max_replicas: Optional[int] = Field(
        None, description="The most replicas the agent may run when it may add replicas: the user's replica limit. "
                          "Default 10.")
    observe: Optional[str] = Field(
        None, description="Only for target_kind 'change': item=state, separated by ';'. A side chain: "
                          "'A:111=t' (states g+, t, g-). A distance: 'A:88:OH-A:111:CG1@0.6=far' (near or far).")
    hold_ps: Optional[float] = Field(None, description="Only for 'change': how long (ps) the new state must last to "
                                                       "count. Default 100.")
    seen_in: Optional[int] = Field(None, description="Only for 'change': in how many replicas it must be seen. "
                                                     "Default 1.")
    distances: Optional[str] = Field(
        None, description="Only for target_kind 'distance': atom pairs with a cutoff in nm, separated by ';', "
                          "for example 'A:88:OH-A:111:CG1@0.6'. Use standard PDB atom names (CA, CB, OH, NZ, "
                          "CG1 ...) and residue numbers from the file summary. Ask if the atoms are unclear.")
    ligand: Optional[str] = Field(None, description="Residue name of the ligand in the PDB file, e.g. BNZ. "
                                                    "Only for target_kind 'ligand'.")
    cutoff_angstrom: Optional[float] = Field(None, description="Distance from the ligand in Å. Default 5.")
    residues: Optional[str] = Field(None, description="Residue list such as 'A:99-120,A:133'. "
                                                      "Only for target_kind 'residues'.")
    forcefield: Optional[Literal["charmm36m", "amber14", "amber19"]] = Field(
        None, description="charmm36m (default): CHARMM36m + CHARMM TIP3P. amber14: ff14SB + TIP3P. "
                          "amber19: ff19SB + OPC.")
    ph: Optional[float] = Field(None, description="Default 7.")
    check_protonation: Optional[bool] = Field(
        None, description="True: predict pKa values with PROPKA and use them. Default false.")
    protonation: Optional[str] = Field(
        None, description="Manual protonation states, e.g. 'A:31=HIP,A:70=ASH'. "
                          "Allowed: ASP/ASH, GLU/GLH, LYS/LYN, HIP/HID/HIE.")
    ion_conc: Optional[float] = Field(None, description="Salt concentration in M. Default 0.15.")
    stability_tolerance_nm: Optional[float] = Field(None, description="Only for 'stability', if the user gives a "
                                                      "precision for the mean RMSD/Rg in nm. Default 0.03.")
    rmsd_cutoff_nm: Optional[float] = Field(None, description="Only for 'stability': the RMSD (nm) above which the "
                                                              "structure counts as departed or unfolded. Default 0.3.")
    max_departure_rate_per_us: Optional[float] = Field(None, description="Only for 'stability', if the user says how "
                                                       "long the structure must stay: 'stable for T' means 1/T per µs "
                                                       "(T = 1 µs -> 1, T = 100 ns -> 10). Default 10.")
    cluster_cutoff_nm: Optional[float] = Field(None, description="RMSD cutoff (nm) of the conformational clustering. "
                                                                 "Default 0.15.")
    cluster: Optional[bool] = Field(None, description="True if the user wants conformational clusters for a run that "
                                                      "is not 'stability' (stability runs always get them).")
    temperature_k: Optional[float] = Field(None, description="Simulation temperature in K, only if the user asks for "
                                                             "a temperature other than the default. Default 300.")
    cation: Optional[Literal["Na+", "K+"]] = Field(None, description="Default Na+. The anion is Cl-.")
    box_shape: Optional[Literal["cube", "dodecahedron", "octahedron"]] = Field(None, description="Default cube.")
    padding_nm: Optional[float] = Field(None, description="Protein to box face distance in nm. Default 1.0.")
    replicas: Optional[int] = Field(None, description="Number of plain MD replicas to start with. Default 3.")
    first_ns: Optional[float] = Field(None, description="Length of round 1 per replica in ns. Default 1.")
    extend_ns: Optional[float] = Field(None, description="Length of each later round in ns. Default: 5, or 1/20 of "
                                                         "max_ns for long budgets.")
    max_ns: Optional[float] = Field(None, description="Limit on the plain MD length per replica in ns. Only if the "
                                                      "user gives one (1 us = 1000 ns). Default: 2000 when MDAgent "
                                                      "plans, 20 for a fixed plan.")
    no_flip_patience_ns: Optional[float] = Field(
        None, description="Only if the user names such a time: give up on plain MD for an item that has not changed "
                          "state after this many ns per replica. Default: decided from the budget.")
    sampling: Optional[Literal["continue", "adaptive"]] = Field(
        None, description="continue (default): each replica continues its own trajectory. "
                          "adaptive: AdaptivePy chooses where each replica restarts.")
    enhanced_ns: Optional[float] = Field(None, description="Length of each enhanced sampling replica in ns. "
                                                           "Default 20.")
    enhanced_replicas: Optional[int] = Field(None, description="Default 2.")
    mode: Optional[Literal["auto", "guided"]] = Field(
        None, description="auto (default): MDAgent chooses the methods, when to switch and when to stop. guided: "
                          "the user named the method (for example 'only plain MD', 'use adaptive sampling'); MDAgent "
                          "runs only that and suggests changes in the chat.")
    goal: Optional[str] = Field(None, description="What the user wants to know, in the user's own words.")
    precision: Optional[float] = Field(
        None, description="Largest allowed error of a state population (0 to 1), only if the user "
                          "states a precision (for example 'within 5%' -> 0.05). Default 0.05.")
    criterion: Optional[Literal["standard_error", "confidence", "sequence"]] = Field(
        None, description="Only if the user asks for a careful or a quick judgement of when to stop. "
                          "confidence or sequence: wider intervals, so it does not stop too early (costs more "
                          "time). Default standard_error.")
    allow: Optional[str] = Field(
        None, description="Only if the user says what MDAgent may do by itself: comma separated from extend, "
                          "add_replicas, switch_method, stop_early, or 'none'. Default: only stop_early in auto mode (the "
                          "time and replica numbers are limits; other changes are suggested and the user confirms), "
                          "none in guided mode.")
    equil_ps: Optional[float] = Field(None, description="Total equilibration before production, in ps. "
                                                        "Default 600.")
    enhanced_max_residues: Optional[int] = Field(
        None, description="Bias at most this many residues at once in enhanced sampling (the slowest ones). "
                          "Default 10.")
    no_enhanced_sampling: Optional[bool] = Field(None, description="True: stop after plain MD.")
    chains: Optional[str] = Field(None, description="Keep only these protein chains, for example 'A' or 'A,B'.")
    keep_tags: Optional[bool] = Field(None, description="True: keep tag residues at the chain ends. Default false "
                                                        "(tags are removed).")
    uniprot: Optional[str] = Field(None, description="UniProt accession of the reference sequence: for files that "
                                                     "have none, or when the structure check says the file's entry is "
                                                     "no longer active (use the one the user chooses).")
    termini: Optional[Literal["auto", "charged", "capped"]] = Field(
        None, description="Chain ends. auto (default): neutral ACE/NME caps where the real protein continues "
                          "beyond the structure, charged ends otherwise. charged: always NH3+/COO-. capped: always "
                          "ACE/NME.")
    accept_problems: Optional[bool] = Field(
        None, description="True only when the user explicitly says to go on despite the problems that the "
                          "structure check reported.")
    questions: list[str] = Field(
        description="Questions for the user when the request is unclear or cannot be done with these "
                    "options. Empty when everything is clear.")
    summary: str = Field(description="One or two sentences, in the user's language, that say what will run.")


SYSTEM = """You turn a researcher's request into settings for MDAgent, a molecular dynamics tool.
MDAgent builds a solvated system from a protein PDB file and runs MD until the quantity the user asks for is
precise enough: side-chain conformations (chi1), distances between atoms, backbone conformations (phi/psi
basins, transitions between them, the phi/psi free-energy landscape, beta-turn types), or structure stability
(backbone RMSD from the start and radius of gyration followed every round, stopping once they are steady and
precise, with the time spent native-like, an RMSD-Rg free-energy surface and conformational clusters). For side chains that
plain MD cannot sample it runs enhanced sampling.

By default MDAgent plans how many replicas, how long and when to stop, for the least compute, within at most 10
replicas of at most 2 us, and stops as soon as the results are precise enough: leave allocation, replicas,
max_ns, max_replicas and allow null unless the user gives them. If the user restricts the method to standard
(unbiased, plain) MD, keep mode null and leave allow null (it may then stop early and add replicas within the
limit, never switch method). Put "extend" in allow only if the user allows going past a limit. Use allocation
"rounds" with mode "guided" only when the user asks for a fixed plan with exact numbers ("exactly 3 replicas of
10 ns").

Defaults (state them when the user's request matches, as "met"; never say a default cannot be set): 300 K,
0.15 M NaCl, CHARMM36m with TIP3P water, pH 7, a cubic box with 1.0 nm of water around the protein. The user may
change the temperature, the salt concentration, the cation (Na+ or K+), the force field, the pH and the box.

Rules:
- Fill in only what the user asked for. Leave every other field null, so it keeps its default.
- Never guess a ligand name, a residue number or a chain. Use only names and numbers that appear in
  the PDB summary below. If the user's words do not match it, ask in `questions`.
- If the request needs something these settings cannot do, say so in `questions`.
- If `questions` is not empty, the run will not start. Ask only what is needed.
- Write `summary` and `questions` in the language of the user's request."""


def make_plan(request: str, summary: str, client=None, backend: str = "api") -> RunPlan:
    """Ask the language model to fill in RunPlan.

    backend "claude" or "codex" uses that login (see backends.py). "api" uses the
    Anthropic API, through `client` if one is given.
    """
    if backend != "api":
        from . import backends

        return backends.call(backend, f"{SYSTEM}\n\nPDB summary:\n{summary}", request, RunPlan)
    import anthropic

    client = client or anthropic.Anthropic()
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # If a safety classifier declines, the API retries on a fallback model.
        output_config={"effort": "low"},
        system=f"{SYSTEM}\n\nPDB summary:\n{summary}",
        messages=[{"role": "user", "content": request}],
        output_format=RunPlan,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise ValueError(f"Claude did not return settings (stop reason: {response.stop_reason}).")
    return response.parsed_output


FLAGS = [  # RunPlan field -> command line option
    ("cutoff_angstrom", "--cutoff-angstrom"), ("forcefield", "--forcefield"), ("ph", "--ph"),
    ("protonation", "--protonation"), ("ion_conc", "--ion-conc"), ("cation", "--cation"),
    ("temperature_k", "--temperature"), ("stability_tolerance_nm", "--stability-tolerance"),
    ("rmsd_cutoff_nm", "--rmsd-cutoff"), ("cluster_cutoff_nm", "--cluster-cutoff"),
    ("max_departure_rate_per_us", "--max-departure-rate"),
    ("hold_ps", "--hold-ps"), ("seen_in", "--seen-in"),
    ("box_shape", "--box-shape"), ("padding_nm", "--padding-nm"), ("replicas", "--replicas"),
    ("first_ns", "--first-ns"), ("extend_ns", "--extend-ns"), ("max_ns", "--max-ns"),
    ("no_flip_patience_ns", "--no-flip-patience-ns"), ("sampling", "--sampling"),
    ("enhanced_ns", "--enhanced-ns"), ("enhanced_replicas", "--enhanced-replicas"),
    ("enhanced_max_residues", "--enhanced-max-residues"), ("equil_ps", "--equil-ps"), ("mode", "--mode"),
    ("goal", "--goal"), ("precision", "--precision"), ("criterion", "--criterion"), ("allow", "--allow"),
    ("allocation", "--allocation"), ("budget_ns", "--budget-ns"), ("max_replicas", "--max-replicas"),
    ("basins", "--basins"), ("rate_rel_error", "--rate-rel-error"), ("fes_error_kcal", "--fes-error-kcal"),
    ("chains", "--chains"), ("uniprot", "--uniprot"), ("termini", "--termini"),
]
SWITCHES = [("check_protonation", "--check-protonation"), ("no_enhanced_sampling", "--no-enhanced-sampling"),
            ("keep_tags", "--keep-tags"), ("accept_problems", "--accept-problems"), ("cluster", "--cluster")]


def to_argv(plan: RunPlan, pdb: str | None, out: str, prmtop: str | None = None,
            inpcrd: str | None = None) -> list[str]:
    """The `mdagent run` command line that matches the plan.

    Input is either a PDB file, or a prepared Amber system (prmtop and inpcrd).
    """
    if prmtop:
        if not inpcrd:
            raise ValueError("A prepared Amber system needs both the prmtop and the inpcrd file.")
        argv = ["run", "--prmtop", str(prmtop), "--inpcrd", str(inpcrd), "--out", str(out)]
    elif pdb:
        argv = ["run", "--pdb", str(pdb), "--out", str(out)]
    else:
        raise ValueError("The plan has no input file.")
    if plan.target_kind == "change":
        if not plan.observe:
            raise ValueError("The plan waits for a change but gives none.")
        argv += ["--observe", plan.observe]
        if plan.residues:
            argv += ["--residues", plan.residues]
    elif plan.target_kind == "distance":
        if not plan.distances:
            raise ValueError("The plan studies distances but gives none.")
        argv += ["--quantity", "distance", "--distances", plan.distances]
        if plan.residues:
            argv += ["--residues", plan.residues]
    elif plan.target_kind == "backbone":
        if plan.backbone:
            argv += ["--quantity", "backbone", "--backbone", plan.backbone]
            if plan.residues:
                argv += ["--residues", plan.residues]
        elif plan.residues or plan.ligand:
            argv += ["--quantity", "backbone"] + (["--residues", plan.residues] if plan.residues
                                                  else ["--around-ligand", plan.ligand])
        else:
            raise ValueError("The plan studies backbone basins but names none.")
    elif plan.target_kind == "stability":
        argv += ["--quantity", "stability"] + (["--residues", plan.residues] if plan.residues else [])
    elif plan.target_kind == "ligand":
        if not plan.ligand:
            raise ValueError("The plan targets a ligand but names none.")
        argv += ["--around-ligand", plan.ligand]
    else:
        if not plan.residues:
            raise ValueError("The plan targets residues but lists none.")
        argv += ["--residues", plan.residues]
    for field_name, flag in FLAGS:
        value = getattr(plan, field_name)
        if value is not None:
            argv += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
    for field_name, flag in SWITCHES:
        if getattr(plan, field_name):
            argv.append(flag)
    return argv


def ask(request: str, pdb: str, out: str, assume_yes: bool = False, planner=None,
        confirm=input, backend: str = "auto") -> int:
    from .cli import build_parser, main

    if not Path(pdb).exists():
        raise ValueError(f"PDB file not found: {pdb}")
    if planner is None:
        from . import backends

        chosen = backends.choose(backend)
        planner = lambda req, summary: make_plan(req, summary, backend=chosen)  # noqa: E731
        print(f"Asking the language model ({chosen}) to turn the request into settings ...")
    try:
        plan = planner(request, pdb_summary(pdb))
    except ImportError as e:
        raise ValueError("The ask command needs the Anthropic SDK: pip install anthropic") from e

    print("\n" + plan.summary)
    if plan.questions:
        print("\nBefore anything runs, please answer:")
        for q in plan.questions:
            print(f"  - {q}")
        print("\nThen run 'mdagent ask' again with a clearer request.")
        return 3

    argv = to_argv(plan, pdb, out)
    try:
        build_parser().parse_args(argv)  # The same checks as the command line.
    except SystemExit:
        print("error: the settings from Claude are not valid options (see the message above).", file=sys.stderr)
        return 2
    changed = [f"{name} = {getattr(plan, name)}" for name, _ in FLAGS + SWITCHES
               if getattr(plan, name) not in (None, False)]
    target = (f"residues within {plan.cutoff_angstrom or 5:g} Å of {plan.ligand}" if plan.target_kind == "ligand"
              else f"backbone basins {plan.backbone}" if plan.target_kind == "backbone"
              else f"the change {plan.observe}" if plan.target_kind == "change"
              else f"distances {plan.distances}" if plan.target_kind == "distance"
              else f"backbone phi/psi of residues {plan.residues or plan.ligand}" if plan.target_kind == "backbone"
              else f"residues {plan.residues}")
    print(f"\nTarget: {target}")
    print("Changed from the defaults: " + (", ".join(changed) if changed else "nothing"))
    print("Command: python -m mdagent " + " ".join(shlex.quote(a) for a in argv))
    if not assume_yes and confirm("\nStart this run? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Not started.")
        return 0
    return main(argv)

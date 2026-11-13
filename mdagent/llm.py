"""Plain-language requests: Claude fills in the run settings, the user confirms.

Claude only fills in RunPlan, a fixed form whose fields are the options of
`mdagent run`. A field left empty keeps its default. The form is turned into
a normal `mdagent run` command line and checked by the same parser as the
command line. The user sees the settings and the command, and confirms
before anything runs. If the request is unclear, Claude asks questions
instead, and nothing runs.

Needs: pip install anthropic, and an API key (ANTHROPIC_API_KEY) or
`ant auth login`.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

MODEL = "claude-opus-5-5"


class RunPlan(BaseModel):
    """Settings for one `mdagent run`. Leave a field null to keep its default."""

    target_kind: Literal["ligand", "residues"] = Field(
        description="'ligand': the target is all protein residues near a ligand in the PDB file. "
                    "'residues': the target is a list of residues.")
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
    cation: Optional[Literal["Na+", "K+"]] = Field(None, description="Default Na+. The anion is Cl-.")
    box_shape: Optional[Literal["cube", "dodecahedron", "octahedron"]] = Field(None, description="Default cube.")
    padding_nm: Optional[float] = Field(None, description="Protein to box face distance in nm. Default 1.0.")
    replicas: Optional[int] = Field(None, description="Number of plain MD replicas. Default 3.")
    first_ns: Optional[float] = Field(None, description="Length of round 1 per replica in ns. Default 1.")
    extend_ns: Optional[float] = Field(None, description="Length of each later round in ns. Default 5.")
    max_ns: Optional[float] = Field(None, description="Total plain MD budget per replica in ns. Default 20.")
    no_flip_patience_ns: Optional[float] = Field(
        None, description="Send a side chain to enhanced sampling if it has not flipped after this many ns. "
                          "Default 10.")
    sampling: Optional[Literal["continue", "adaptive"]] = Field(
        None, description="continue (default): each replica continues its own trajectory. "
                          "adaptive: AdaptivePy chooses where each replica restarts.")
    enhanced_ns: Optional[float] = Field(None, description="Length of each enhanced sampling replica in ns. "
                                                           "Default 20.")
    enhanced_replicas: Optional[int] = Field(None, description="Default 2.")
    no_enhanced_sampling: Optional[bool] = Field(None, description="True: stop after plain MD.")
    questions: list[str] = Field(
        description="Questions for the user when the request is unclear or cannot be done with these "
                    "options. Empty when everything is clear.")
    summary: str = Field(description="One or two sentences, in the user's language, that say what will run.")


SYSTEM = """You turn a researcher's request into settings for MDAgent, a molecular dynamics tool.
MDAgent builds a solvated system from a protein PDB file, runs plain MD in rounds, finds side chains
in a target region that did not sample their chi1 states, and runs enhanced sampling on them.

Rules:
- Fill in only what the user asked for. Leave every other field null, so it keeps its default.
- Never guess a ligand name, a residue number or a chain. Use only names and numbers that appear in
  the PDB summary below. If the user's words do not match it, ask in `questions`.
- If the request needs something these settings cannot do, say so in `questions`.
- If `questions` is not empty, the run will not start. Ask only what is needed.
- Write `summary` and `questions` in the language of the user's request."""


def pdb_summary(pdb_path: str | Path) -> str:
    """Chains, residue ranges and non-protein molecules in the PDB file, for the model."""
    import mdtraj as md

    top = md.load_topology(str(pdb_path))
    lines = []
    for chain in top.chains:
        prot = [r for r in chain.residues if r.is_protein]
        if prot:
            lines.append(f"Chain {chain.chain_id}: protein residues {prot[0].resSeq}-{prot[-1].resSeq} "
                         f"({len(prot)} residues)")
    others: dict[str, int] = {}
    for r in top.residues:
        if not r.is_protein and not r.is_water:
            others[r.name] = others.get(r.name, 0) + 1
    if others:
        lines.append("Non-protein molecules (residue name x count): "
                     + ", ".join(f"{k} x{v}" for k, v in sorted(others.items())))
    else:
        lines.append("No ligands or other non-protein molecules.")
    return "\n".join(lines)


def make_plan(request: str, summary: str, client=None) -> RunPlan:
    """Ask Claude to fill in RunPlan."""
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
    ("box_shape", "--box-shape"), ("padding_nm", "--padding-nm"), ("replicas", "--replicas"),
    ("first_ns", "--first-ns"), ("extend_ns", "--extend-ns"), ("max_ns", "--max-ns"),
    ("no_flip_patience_ns", "--no-flip-patience-ns"), ("sampling", "--sampling"),
    ("enhanced_ns", "--enhanced-ns"), ("enhanced_replicas", "--enhanced-replicas"),
]
SWITCHES = [("check_protonation", "--check-protonation"), ("no_enhanced_sampling", "--no-enhanced-sampling")]


def to_argv(plan: RunPlan, pdb: str, out: str) -> list[str]:
    """The `mdagent run` command line that matches the plan."""
    argv = ["run", "--pdb", str(pdb), "--out", str(out)]
    if plan.target_kind == "ligand":
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


def ask(request: str, pdb: str, out: str, assume_yes: bool = False, planner=make_plan,
        confirm=input) -> int:
    from .cli import build_parser, main

    if not Path(pdb).exists():
        raise ValueError(f"PDB file not found: {pdb}")
    print("Asking Claude to turn the request into settings ...")
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
              else f"residues {plan.residues}")
    print(f"\nTarget: {target}")
    print("Changed from the defaults: " + (", ".join(changed) if changed else "nothing"))
    print("Command: python -m mdagent " + " ".join(shlex.quote(a) for a in argv))
    if not assume_yes and confirm("\nStart this run? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Not started.")
        return 0
    return main(argv)

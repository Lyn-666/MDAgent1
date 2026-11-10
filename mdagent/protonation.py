"""Protonation states: prediction with PROPKA and manual choices.

By default, Asp, Glu and Lys are charged, and His is neutral (OpenMM puts
the hydrogen on ND1 or NE2, whichever makes a better hydrogen bond).
These rules ignore the protein environment. PROPKA predicts the pKa of each
residue from its environment. When the predicted pKa puts a residue in a
different state at the chosen pH, we use the predicted state.

State names are OpenMM residue variant names:
  ASP (charged) / ASH (neutral)      GLU (charged) / GLH (neutral)
  LYS (charged) / LYN (neutral)      HIP (charged) / HID or HIE (neutral)
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openmm import app

ResidueKey = tuple[str, int]

ALLOWED = {
    "ASP": ("ASP", "ASH"),
    "GLU": ("GLU", "GLH"),
    "LYS": ("LYS", "LYN"),
    "HIS": ("HIP", "HID", "HIE"),
}
CHARGED = {"ASP": "ASP", "GLU": "GLU", "LYS": "LYS", "HIS": "HIP"}
NEUTRAL = {"ASP": "ASH", "GLU": "GLH", "LYS": "LYN", "HIS": None}  # None: OpenMM picks HID or HIE.
ACIDS = {"ASP", "GLU"}


@dataclass
class PkaPrediction:
    chain: str
    resseq: int
    resname: str
    pka: float
    default_charged: bool
    predicted_charged: bool

    @property
    def changed(self) -> bool:
        return self.default_charged != self.predicted_charged

    @property
    def variant(self) -> str | None:
        return CHARGED[self.resname] if self.predicted_charged else NEUTRAL[self.resname]


def _charged_at(resname: str, pka: float, ph: float) -> bool:
    # An acid is charged when it has lost its proton (pH above pKa).
    # A base is charged when it holds the proton (pH below pKa).
    return ph > pka if resname in ACIDS else ph < pka


def predict(topology: app.Topology, positions, ph: float) -> list[PkaPrediction]:
    """Run PROPKA on a heavy-atom structure. Return Asp, Glu, Lys and His only."""
    import propka.run

    logging.getLogger("propka").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "protein.pdb"
        with open(path, "w") as f:
            app.PDBFile.writeFile(topology, positions, f, keepIds=True)
        mol = propka.run.single(str(path), optargs=["--quiet"], write_pka=False)

    out = []
    for g in mol.conformations["AVR"].groups:
        name = g.residue_type
        if name not in ALLOWED:
            continue
        # PROPKA calls the carboxyl group of Asp and Glu "COO". The C-terminus
        # is also "COO", but its residue_type is "C-", so it is already skipped.
        if g.type not in ("COO", "LYS", "HIS"):
            continue
        default_charged = name != "HIS"
        out.append(PkaPrediction(
            chain=g.atom.chain_id, resseq=int(g.atom.res_num), resname=name,
            pka=float(g.pka_value), default_charged=default_charged,
            predicted_charged=_charged_at(name, float(g.pka_value), ph),
        ))
    return sorted(out, key=lambda p: (p.chain, p.resseq))


_ITEM = re.compile(r"^(?:(?P<chain>[A-Za-z0-9]):)?(?P<num>-?\d+)=(?P<state>[A-Za-z]{3})$")


def parse_overrides(spec: str) -> dict[tuple[str | None, int], str]:
    """Parse a list such as "A:31=HIP,70=ASH". A missing chain ID means any chain."""
    out = {}
    for item in spec.replace(" ", "").split(","):
        if not item:
            continue
        m = _ITEM.match(item)
        if not m:
            raise ValueError(f"Cannot parse protonation item {item!r}. Use a form like 'A:31=HIP'.")
        out[(m["chain"], int(m["num"]))] = m["state"].upper()
    return out


def format_predictions(preds: list[PkaPrediction], ph: float, target: set[ResidueKey]) -> str:
    changed = [p for p in preds if p.changed]
    lines = [f"PROPKA checked {len(preds)} Asp, Glu, Lys and His residues at pH {ph:g}."]
    if not changed:
        lines.append("All of them keep the default state.")
        return "\n".join(lines)
    lines.append(f"{len(changed)} residue(s) change state because of the protein environment:")
    for p in changed:
        state = "charged" if p.predicted_charged else "neutral"
        where = " (in the target region)" if (p.chain, p.resseq) in target else ""
        lines.append(f"  {p.resname}{p.resseq} chain {p.chain}: predicted pKa {p.pka:.1f}, "
                     f"so it is {state} at pH {ph:g}{where}.")
    return "\n".join(lines)

"""Where the results of a run are, so that the user can find them.

The files are on the computer that runs the simulation. If the chat page is
opened on another computer (for example through SSH port forwarding), that
computer does not have the files. So every message about output files names
the computer and gives the full path, and the chat page offers downloads.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

# Result files, as glob patterns relative to the run folder, with a short description.
# The plain MD pattern also matches the full-system files of adaptive sampling (.full.dcd).
OUTPUTS = [
    ("report.*.md", "result report: results with errors, stop reasons, actions, limits (Chinese and English)"),
    ("status.json", "latest plain MD diagnosis: populations, errors and verdict of every item"),
    ("landscape_*.svg", "phi/psi free-energy landscape of a backbone observable (figure)"),
    ("landscape_*.csv", "phi/psi free-energy landscape of a backbone observable (every 10 degree bin)"),
    ("protein.pdb", "topology for all .dcd files (protein atoms only)"),
    ("replica_*_part*.dcd", "plain MD trajectory of replica i, round k"),
    ("enhanced/replica_*_chunk*.dcd", "enhanced sampling trajectory of replica i, chunk k"),
    ("enhanced/replica_*_chi.npy", "the biased dihedrals (radians; chi1, or phi and psi of each residue), one row "
                                   "per pace"),
    ("enhanced/replica_*_bias.npy", "final bias of each biased dihedral on its grid (kJ/mol)"),
    ("backbone/*_fes.csv", "phi/psi free-energy landscape of a residue: F and its error per bin (kcal/mol)"),
    ("backbone/*_fes_reweighted.csv", "the same from enhanced sampling, reweighted (populations come from it)"),
    ("backbone/*_fes_reweighted.npz", "the reweighted landscape as numpy arrays"),
    ("backbone/*_fes.npz", "the same landscape as numpy arrays (edges_deg, free_energy_kcal, error_kcal, probability)"),
    ("backbone/*_fes.png", "a picture of the landscape (only if matplotlib is installed)"),
    ("backbone/*.json", "transition counts, rates and dwell times between basins, and the lowest F in each basin"),
    ("enhanced/populations.json", "reweighted state populations and free energies of the biased residues"),
    ("stability/rmsd_rg.svg", "RMSD from the start and radius of gyration against time (figure)"),
    ("stability/rmsd_rg.csv", "RMSD and radius of gyration against time, per replica (nm)"),
    ("stability/fes_rmsd_rg.svg", "free-energy surface F(RMSD, Rg) (figure)"),
    ("stability/fes_rmsd_rg.csv", "free-energy surface F(RMSD, Rg) with errors per bin (kcal/mol)"),
    ("stability/summary.json", "structure stability: means, errors, steady state, time native-like"),
    ("clusters/cluster_*.pdb", "representative structure of each conformational cluster"),
    ("clusters/clusters.json", "conformational clusters: populations with errors, replicas, centres"),
]


def host() -> str:
    """Name of this computer."""
    return socket.gethostname()


def _natural(path: Path) -> list:
    """Sort key that puts replica_2 before replica_10."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(path))]


def output_files(run_dir: Path) -> list[Path]:
    """The result files that exist now, relative to the run folder, in a stable order."""
    run_dir = Path(run_dir)
    found = []
    for pattern, _ in OUTPUTS:
        found += sorted((p.relative_to(run_dir) for p in run_dir.glob(pattern) if p.is_file()), key=_natural)
    return found


def describe(run_dir: Path) -> list[str]:
    """Lines for the end of a run log: the computer, the folder, and what each kind of file holds."""
    run_dir = Path(run_dir).resolve()
    files = output_files(run_dir)
    lines = [f"Output files are on computer '{host()}', in folder {run_dir}"]
    for pattern, text in OUTPUTS:
        n = sum(1 for f in files if f.match(pattern))
        if n:
            shown = pattern.replace("replica_*_part*", "replica_<i>_part<k>") \
                           .replace("replica_*_chunk*", "replica_<i>_chunk<k>").replace("replica_*", "replica_<i>")
            lines.append(f"  {shown:36s} {text} ({n} file{'s' if n > 1 else ''})")
    return lines

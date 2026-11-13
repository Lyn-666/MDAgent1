"""Download protein structures from the RCSB Protein Data Bank.

Files go to structures/<ID>.pdb in the project folder, for example
structures/3HTB.pdb, so they are easy to find. A file that is already there is
not downloaded again.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

FOLDER = Path("structures")
URL = "https://files.rcsb.org/download/{id}.pdb"
_ID = re.compile(r"^[1-9][A-Za-z0-9]{3}$")


def normalize_id(pdb_id: str) -> str:
    pid = pdb_id.strip().upper()
    if not _ID.match(pid):
        raise ValueError(f"'{pdb_id}' is not a PDB ID. A PDB ID has four characters and starts with a digit, "
                         "for example 181L or 3HTB.")
    return pid


def pdb_path(pdb_id: str, root: Path = Path(".")) -> Path:
    return root / FOLDER / f"{normalize_id(pdb_id)}.pdb"


def fetch_pdb(pdb_id: str, root: Path = Path("."), opener=None) -> tuple[Path, bool]:
    """Download the structure if needed. Returns (path, downloaded_now)."""
    opener = opener or urllib.request.urlopen
    pid = normalize_id(pdb_id)
    path = pdb_path(pid, root)
    if path.exists():
        return path, False
    try:
        with opener(URL.format(id=pid), timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"The PDB has no file in PDB format for {pid}. The ID may be wrong, or the entry may "
                             "exist only in mmCIF format (large structures), which MDAgent cannot read yet.") from e
        raise ValueError(f"Download of {pid} failed: HTTP {e.code}.") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Download of {pid} failed: {e.reason}. Check the internet connection.") from e
    if b"ATOM" not in data and b"HETATM" not in data:
        raise ValueError(f"The file downloaded for {pid} has no atoms.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)  # The file appears only when it is complete.
    return path, True

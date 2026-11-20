"""The general core must not depend on proteins, MD engines, test systems or task-specific modules."""

import ast
import re
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "mdagent" / "core"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


def test_core_imports_only_the_standard_library_and_itself():
    stdlib = set(sys.stdlib_module_names)
    for path in CORE.glob("*.py"):
        for name in _imports(path):
            if name.startswith("."):
                assert name.count(".") == 1 or name == ".", f"{path.name} reaches outside the core: {name}"
                continue
            assert name.split(".")[0] in stdlib, f"{path.name} imports {name}, which is not standard library"


def test_core_names_no_observable_system_or_engine():
    banned = re.compile(r"chi1|rotamer|dihedral|ramachandran|phi/psi|openmm|mdtraj|plumed|metadynamics|"
                        r"1UBQ|181L|ubiquitin|lysozyme|residue", re.I)
    for path in CORE.glob("*.py"):
        hits = banned.findall(path.read_text())
        assert not hits, f"{path.name} mentions task-specific terms: {sorted(set(hits))}"

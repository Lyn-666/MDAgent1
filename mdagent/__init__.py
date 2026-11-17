"""An MD agent that finds and fixes poorly sampled side chains in a target region."""

import warnings

__version__ = "0.1.0"

# NMR structures (for example 1ZDD) have a placeholder unit cell in the PDB file. mdtraj warns and
# ignores it, which is the right thing to do; the warning only confuses users. MDAgent builds its own box.
warnings.filterwarnings("ignore", message="Unlikely unit cell vectors", category=UserWarning)

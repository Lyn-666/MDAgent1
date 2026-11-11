"""Check the OpenMM installation before a run, and say how to fix problems."""

from __future__ import annotations

import os

import openmm
from openmm import app

GPU_PLATFORMS = ("CUDA", "HIP", "OpenCL")
REINSTALL = 'conda install -c conda-forge "openmm>=8.6"'


def available_platforms() -> list[str]:
    return [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]


def fastest_platform() -> str:
    platforms = [openmm.Platform.getPlatform(i) for i in range(openmm.Platform.getNumPlatforms())]
    return max(platforms, key=lambda p: p.getSpeed()).getName()


def check_environment(platform: str | None = None, forcefield_files: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Return (notes, problems). Problems stop the run; notes are for information."""
    names = available_platforms()
    notes, problems = [], []
    chosen = platform or fastest_platform()
    notes.append(f"OpenMM {openmm.__version__}. Platforms: {', '.join(names)}. Using {chosen}.")

    if platform and platform not in names:
        problems.append(f"The platform {platform} is not available. Available: {', '.join(names)}.")
    if not any(p in names for p in GPU_PLATFORMS):
        notes.append("WARNING: no GPU platform was found. MD on the CPU is very slow. "
                     f"If this computer has a GPU, reinstall OpenMM: {REINSTALL}")
    elif "CUDA" not in names and "HIP" not in names:
        notes.append("WARNING: the CUDA platform is missing, so OpenMM uses OpenCL, which is usually "
                     f"slower on NVIDIA GPUs. To get CUDA back: {REINSTALL}")

    data = os.path.join(os.path.dirname(app.__file__), "data")
    missing = [f for f in forcefield_files if not os.path.exists(os.path.join(data, f))]
    if missing:
        problems.append(f"This OpenMM version ({openmm.__version__}) does not include {', '.join(missing)}. "
                        f"Update OpenMM ({REINSTALL}) or choose another --forcefield.")
    return notes, problems

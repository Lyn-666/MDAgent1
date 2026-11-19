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


def try_platform(name: str) -> str | None:
    """Compute forces on a two-atom system with this platform. Return the error text, or None if it works.

    A platform can be listed and still fail, for example CUDA when the CUDA parts that conda installed are
    newer than the NVIDIA driver (CUDA_ERROR_UNSUPPORTED_PTX_VERSION). Without this test the run would build
    the whole system first and then fail with a hard-to-read error.
    """
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.1, 1000.0)
    system.addForce(bond)
    try:
        context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(name))
        context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(0.12, 0, 0)])
        context.getState(getForces=True)
        del context
    except Exception as exc:  # OpenMMException, or any error from the GPU driver
        return str(exc).strip() or type(exc).__name__
    return None


def platform_fix(name: str, error: str, names: list[str]) -> str:
    """Say how to fix a platform that is listed but cannot compute forces."""
    text = f"The {name} platform cannot compute forces: {error}"
    if name == "CUDA" and ("PTX" in error or "CUDA_ERROR_UNSUPPORTED" in error):
        text += ("\nThe CUDA parts that conda installed are newer than the NVIDIA driver on this computer. "
                 "Run nvidia-smi and read 'CUDA Version' in the top line (for example 12.4). Then, in the "
                 "mdagent environment, install CUDA parts that match the driver, for example:\n"
                 "    conda install -c conda-forge cuda-version=12.4\n"
                 "and check again with: python -m openmm.testInstallation")
    others = [p for p in GPU_PLATFORMS if p in names and p != name]
    if others:
        text += (f"\nUntil then, you can run on another GPU platform with --platform {others[0]} "
                 "(OpenCL is usually slower than CUDA on NVIDIA GPUs).")
    return text


def check_environment(platform: str | None = None, forcefield_files: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Return (notes, problems). Problems stop the run; notes are for information."""
    names = available_platforms()
    notes, problems = [], []
    chosen = platform or fastest_platform()
    notes.append(f"OpenMM {openmm.__version__}. Platforms: {', '.join(names)}. Using {chosen}.")

    if platform and platform not in names:
        problems.append(f"The platform {platform} is not available. Available: {', '.join(names)}.")
    else:
        error = try_platform(chosen)
        if error:
            problems.append(platform_fix(chosen, error, names))
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

#!/usr/bin/env bash
# Install MDAgent, or repair an existing install, with CUDA parts that match this computer's NVIDIA driver.
#
#   bash scripts/install.sh
#
# Why: conda only checks the major CUDA version of the driver. Left alone it installs the newest CUDA parts
# (for example 13.5 on a driver that supports 13.3, or a 12.9 build on a 12.4 driver), and OpenMM then fails on
# the GPU with errors such as CUDA_ERROR_UNSUPPORTED_PTX_VERSION. This script reads the CUDA version the driver
# supports (nvidia-smi) and asks conda for the newest CUDA parts that are not newer than that.
# If the environment exists, it is repaired in place; otherwise it is created from environment.yml.
# Options (environment variables): MDAGENT_ENV (default mdagent).
set -euo pipefail

ENV_NAME="${MDAGENT_ENV:-mdagent}"
MIN_CUDA="12.6"        # the oldest CUDA that an OpenMM build MDAgent works with (OpenMM >= 8.3) needs
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$*"; }

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniforge (https://github.com/conda-forge/miniforge), open a new terminal,"
    echo "and run this script again."
    exit 1
fi

# --- 1. The CUDA version the NVIDIA driver supports ("CUDA Version" or, in newer drivers, "CUDA UMD Version").
cuda=""
if command -v nvidia-smi >/dev/null 2>&1; then
    cuda="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA( UMD)? Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+$' | head -n 1 || true)"
fi

version_ge() {   # version_ge A B: true when A >= B (major.minor)
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]
}

pin=""
if [ -z "$cuda" ]; then
    say "No NVIDIA driver found (nvidia-smi is missing or failed)."
    echo "MDAgent is installed without a CUDA match. That is fine for short tests on the CPU, but real runs need"
    echo "an NVIDIA GPU with its driver."
elif ! version_ge "$cuda" "$MIN_CUDA"; then
    say "The NVIDIA driver supports CUDA $cuda, but MDAgent's OpenMM needs a driver that supports CUDA $MIN_CUDA or newer."
    echo "Please update the NVIDIA driver (a driver from 2025 or later supports CUDA 12.9 or 13), reboot,"
    echo "and run this script again. Without the update OpenMM cannot use this GPU with CUDA."
    exit 1
else
    major="${cuda%%.*}"
    pin="cuda-version>=${major}.0,<=${cuda}"
    say "The NVIDIA driver supports CUDA $cuda: installing CUDA parts not newer than that ($pin)."
fi

# --- 2. Prefer the fast libmamba solver when conda has it (conda 23.10 and later use it by default).
if conda list -n base 2>/dev/null | grep -q '^conda-libmamba-solver'; then
    export CONDA_SOLVER=libmamba
fi

# --- 3. Create the environment, or repair the existing one.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    say "The environment '$ENV_NAME' exists: matching its CUDA parts and OpenMM to this driver."
    if [ -n "$pin" ]; then
        conda install -y -n "$ENV_NAME" -c conda-forge "$pin" "openmm>=8.3"
    else
        conda install -y -n "$ENV_NAME" -c conda-forge "openmm>=8.3"
    fi
else
    say "Creating the environment '$ENV_NAME' from environment.yml."
    spec="$(mktemp --suffix=.yml)"
    trap 'rm -f "$spec"' EXIT
    if [ -n "$pin" ]; then
        # The same file, with the CUDA pin added as the first dependency.
        awk -v pin="$pin" '{print} /^dependencies:/ {print "  - " pin}' environment.yml > "$spec"
    else
        cp environment.yml "$spec"
    fi
    conda env create -n "$ENV_NAME" -f "$spec"
fi

# --- 4. MDAgent itself, then a check that OpenMM really runs on the GPU.
say "Installing MDAgent into '$ENV_NAME'."
conda run -n "$ENV_NAME" python -m pip install -e . --quiet

say "Checking OpenMM."
conda run -n "$ENV_NAME" python -m openmm.testInstallation || true
if [ -n "$pin" ]; then
    if conda run -n "$ENV_NAME" python -c "
import sys
from mdagent.envcheck import try_platform
error = try_platform('CUDA')
print('CUDA works.' if error is None else 'CUDA does not work: ' + error)
sys.exit(0 if error is None else 1)"; then
        say "Done. Start MDAgent with:  conda activate $ENV_NAME && mdagent start"
    else
        say "OpenMM is installed, but CUDA does not work yet. Please send this whole output to the person who invited you."
        exit 1
    fi
else
    say "Done (without CUDA). Start MDAgent with:  conda activate $ENV_NAME && mdagent start"
fi

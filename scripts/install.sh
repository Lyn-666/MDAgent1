#!/usr/bin/env bash
# Install MDAgent, or repair an existing install, with fixed CUDA and OpenMM versions that this computer can run.
#
#   bash scripts/install.sh           install, or repair an existing install
#   bash scripts/install.sh --check   only report what this computer supports and what would be installed
#
# Everyone gets one of two fixed combinations (the user's decision, 2026-11-25), so results are reproducible and
# problems are easy to trace:
#   A. driver supports CUDA 12.9 or newer (all CUDA 13 drivers too):  cuda-version 12.9 + OpenMM 8.6.1
#   B. driver supports CUDA 12.6 to 12.8:                              cuda-version 12.6 or 12.8 + OpenMM 8.4.0
#   below 12.6: the driver must be updated.
# Why fixed CUDA 12 and not the newest: a driver runs every CUDA version up to the one it reports, but not newer
# ones; CUDA 12.9 runs on all newer drivers, and unlike CUDA 13 it still supports GPUs from Maxwell (2014) to
# Blackwell (RTX 50). Left alone, conda only checks the major version and installs parts newer than the driver
# (for example 13.5 on a 13.3 driver), and OpenMM then fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
# If the environment exists, it is repaired in place; otherwise it is created from environment.yml.
# Last, the language model for the chat: Claude Code (claude) or Codex (codex), installed into the environment with
# its own Node.js (no admin rights, no PATH changes) unless already on this computer, and logged in. Login works
# without a browser on this computer: it shows a link to open on any computer or phone.
# Options (environment variables): MDAGENT_ENV (default mdagent); MDAGENT_LLM = claude, codex, both or none
# (otherwise the script asks).
set -euo pipefail

ENV_NAME="${MDAGENT_ENV:-mdagent}"
MIN_CUDA="12.6"           # combination B needs at least this
A_CUDA="12.9"; A_OPENMM="8.6.1"
B_OPENMM="8.4.0"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$*"; }

if [ "$CHECK_ONLY" = 0 ] && ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found. Install Miniforge (https://github.com/conda-forge/miniforge), open a new terminal,"
    echo "and run this script again."
    exit 1
fi

version_ge() {   # version_ge A B: true when A >= B (major.minor)
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]
}

# --- 1. What the computer has: the CUDA version the driver supports ("CUDA Version", or "CUDA UMD Version" in
# newer drivers), and the GPUs with their compute capability (with several GPUs, the oldest one counts).
cuda=""
gpus=""
cc_min=""
if command -v nvidia-smi >/dev/null 2>&1; then
    cuda="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA( UMD)? Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+$' | head -n 1 || true)"
    gpus="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || true)"
    cc_min="$(printf '%s\n' "$gpus" | awk -F', *' 'NF >= 2 && $2 ~ /^[0-9]+\.[0-9]+$/ {print $2}' | sort -V | head -n 1)"
    if [ -n "$gpus" ]; then
        say "GPUs (name, compute capability, driver):"
        printf '%s\n' "$gpus" | sed 's/^/    /'
    fi
fi

# --- 2. Choose the combination.
cuda_pin="cuda-version=$A_CUDA"
openmm_pin="openmm=$A_OPENMM"
use_cuda=1
if [ -z "$cuda" ]; then
    use_cuda=0
    say "No NVIDIA driver found (nvidia-smi is missing or failed)."
    echo "MDAgent installs combination A (CUDA $A_CUDA, OpenMM $A_OPENMM) without a GPU. That is fine for short tests"
    echo "on the CPU, but real runs need an NVIDIA GPU with its driver."
elif ! version_ge "$cuda" "$MIN_CUDA"; then
    say "The NVIDIA driver supports CUDA $cuda, but MDAgent needs a driver that supports CUDA $MIN_CUDA or newer."
    echo "Please update the NVIDIA driver (driver 575 or newer supports CUDA 12.9), reboot, and run this script again."
    echo "Without the update OpenMM cannot use this GPU with CUDA."
    exit 1
elif [ -n "$cc_min" ] && ! version_ge "$cc_min" "5.0"; then
    use_cuda=0
    say "This GPU (compute capability $cc_min) is older than CUDA 12 supports, so OpenMM cannot use it with CUDA."
    echo "MDAgent installs combination A and will try the OpenCL platform instead (slower)."
elif version_ge "$cuda" "$A_CUDA"; then
    say "The NVIDIA driver supports CUDA $cuda: installing combination A (CUDA $A_CUDA, OpenMM $A_OPENMM)."
else
    # CUDA 12.7 was never released (drivers of the 565 series report it), so take the newest of 12.6 and 12.8
    # that the driver can run.
    cuda_pin="cuda-version>=$MIN_CUDA,<=$cuda"
    openmm_pin="openmm=$B_OPENMM"
    say "The NVIDIA driver supports CUDA $cuda: installing combination B (CUDA 12.6 or 12.8, OpenMM $B_OPENMM)."
    echo "Combination A (CUDA $A_CUDA, OpenMM $A_OPENMM) would need driver 575 or newer; it is optional."
fi

if [ "$CHECK_ONLY" = 1 ]; then
    say "Check only: nothing was installed. Run 'bash scripts/install.sh' to install."
    exit 0
fi

# --- 3. Prefer the fast libmamba solver when conda has it (conda 23.10 and later use it by default).
if conda list -n base 2>/dev/null | grep -q '^conda-libmamba-solver'; then
    export CONDA_SOLVER=libmamba
fi

# --- 4. Create the environment, or repair the existing one.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    say "The environment '$ENV_NAME' exists: setting it to $cuda_pin and $openmm_pin."
    conda install -y -n "$ENV_NAME" -c conda-forge "$cuda_pin" "$openmm_pin"
else
    say "Creating the environment '$ENV_NAME' from environment.yml ($cuda_pin, $openmm_pin)."
    spec="$(mktemp --suffix=.yml)"
    trap 'rm -f "$spec"' EXIT
    # environment.yml holds combination A; replace its two pins with the chosen ones.
    sed -e "s|^  - cuda-version=.*|  - $cuda_pin|" -e "s|^  - openmm=.*|  - $openmm_pin|" environment.yml > "$spec"
    conda env create -n "$ENV_NAME" -f "$spec"
fi

# --- 5. MDAgent itself, then a check that OpenMM really runs on the GPU.
say "Installing MDAgent into '$ENV_NAME'."
conda run -n "$ENV_NAME" python -m pip install -e . --quiet

say "Checking OpenMM."
conda run -n "$ENV_NAME" python -m openmm.testInstallation || true
platform_ok() {
    conda run -n "$ENV_NAME" python -c "
import sys
from mdagent.envcheck import try_platform
error = try_platform('$1')
print('$1 works.' if error is None else '$1 does not work: ' + error)
sys.exit(0 if error is None else 1)"
}
gpu_ok=1
if [ "$use_cuda" = 1 ] && platform_ok CUDA; then
    :
elif [ -n "$cuda" ]; then
    gpu_ok=0
    say "OpenMM is installed, but CUDA does not work on this computer."
    if platform_ok OpenCL; then
        echo "The OpenCL platform works on this GPU (slower than CUDA). Short tests can use it:"
        echo "    mdagent run ... --platform OpenCL"
    fi
    echo "Please send this whole output and the output of nvidia-smi to the person who invited you."
fi

# --- 6. The language model for the chat. Its program lives in the environment's bin folder (on PATH after
# 'conda activate'), unless it is already on this computer.
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
BIN="$ENV_PREFIX/bin"
exe() { if [ -x "$BIN/$1" ]; then echo "$BIN/$1"; else command -v "$1" 2>/dev/null || true; fi; }
logged_in() {   # claude auth status / codex login status: exit 0 when logged in
    local path; path="$(exe "$1")"
    [ -n "$path" ] || return 1
    if [ "$1" = claude ]; then env -u ANTHROPIC_API_KEY "$path" auth status >/dev/null 2>&1
    else "$path" login status >/dev/null 2>&1; fi
}
install_cli() {
    if [ -n "$(exe "$1")" ]; then return 0; fi
    if [ ! -x "$BIN/npm" ] || ! "$BIN/node" -e 'process.exit(+process.versions.node.split(".")[0] >= 22 ? 0 : 1)' 2>/dev/null; then
        say "Installing Node.js into '$ENV_NAME' (needed by the $1 program)."
        conda install -y -n "$ENV_NAME" -c conda-forge "nodejs>=22"
    fi
    say "Installing $1 into '$ENV_NAME'."
    if [ "$1" = claude ]; then "$BIN/npm" install -g --prefix "$ENV_PREFIX" @anthropic-ai/claude-code
    else "$BIN/npm" install -g --prefix "$ENV_PREFIX" @openai/codex; fi
}
log_in() {
    if logged_in "$1"; then echo "$1 is logged in."; return 0; fi
    if [ ! -t 0 ]; then
        echo "$1 is not logged in. Log in later with:  conda activate $ENV_NAME && $2"
        return 0
    fi
    say "Log in to $1 with your own account. A link appears: open it in a browser on any computer or phone,"
    echo "sign in, and paste the code back here if you are asked for one."
    if [ "$1" = claude ]; then env -u ANTHROPIC_API_KEY "$(exe claude)" auth login || true
    else "$(exe codex)" login --device-auth || true; fi
    if logged_in "$1"; then echo "$1 is logged in."
    else echo "$1 is not logged in yet. Try again later with:  conda activate $ENV_NAME && $2"; fi
}

choice="${MDAGENT_LLM:-}"
if [ -z "$choice" ]; then
    found=""
    [ -n "$(exe claude)" ] && found="claude"
    [ -n "$(exe codex)" ] && found="${found:+$found }codex"
    if [ -n "$found" ]; then
        choice="$found"
        say "Language model already on this computer: $found."
    elif [ -t 0 ]; then
        say "The chat needs one language model, with your own account (no API key needed):"
        echo "  1) Claude   (Claude Pro, Max, Team or Enterprise plan)"
        echo "  2) ChatGPT  (ChatGPT Plus, Pro, Team or Enterprise plan; the program is called Codex)"
        echo "  3) both"
        echo "  4) skip (I will install one myself)"
        read -r -p "Choose 1, 2, 3 or 4: " answer
        case "$answer" in 1) choice=claude ;; 2) choice=codex ;; 3) choice="claude codex" ;; *) choice=none ;; esac
    else
        choice=none
    fi
fi
[ "$choice" = both ] && choice="claude codex"
llm_ok=0
for name in $choice; do
    case "$name" in
        claude) install_cli claude && log_in claude "claude auth login" ;;
        codex)  install_cli codex && log_in codex "codex login --device-auth" ;;
        *) continue ;;
    esac
    logged_in "$name" && llm_ok=1
done

if [ "$llm_ok" = 0 ]; then
    say "No logged-in language model yet: the chat (mdagent start) needs Claude Code or Codex."
    echo "Run this script again and choose one, or log in later as shown above."
fi
if [ "$gpu_ok" = 1 ] && [ "$llm_ok" = 1 ]; then
    say "Done. Start MDAgent with:  conda activate $ENV_NAME && mdagent start"
else
    exit 1
fi

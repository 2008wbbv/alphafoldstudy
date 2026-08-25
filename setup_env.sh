#!/bin/bash
# =============================================================================
# setup_env.sh
#
# One command to get from a fresh clone to a working environment.
#
#   bash setup_env.sh              install Miniconda if needed, build the env
#   bash setup_env.sh --download   also pre-download every structure
#   bash setup_env.sh --help       show options
#
# Safe to re-run. Anything already installed is detected and reused.
#
# On a cluster, run this on a LOGIN node. Compute nodes usually have no internet,
# so both the package install and the structure downloads have to happen on a
# node that does; the batch jobs then only read from shared storage.
#
# -----------------------------------------------------------------------------
# EDIT THESE IF YOUR CLUSTER NEEDS IT. The defaults work on most machines.
# -----------------------------------------------------------------------------

# Name of the environment to create.
ENV_NAME="${ENV_NAME:-alphafold-study}"

# Where Miniconda gets installed IF this script has to install it.
# Home directories on clusters are often small, and conda plus PyTorch needs
# roughly 5 GB, so point this at scratch or project space if $HOME has a quota.
#   export CONDA_ROOT=/scratch/$USER/miniconda3
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"

# Where package archives and pip wheels are cached. Same quota warning.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CONDA_ROOT/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

# Module that provides conda on an HPC system, if there is one. Run
# "module avail conda" to find the name. Leave empty to skip module loading.
CONDA_MODULE="${CONDA_MODULE:-}"

# =============================================================================
# Nothing below here normally needs editing.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DO_DOWNLOAD=0
for arg in "$@"; do
    case "$arg" in
        --download) DO_DOWNLOAD=1 ;;
        --help|-h)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

echo "=============================================="
echo " AlphaFold study environment setup"
echo "   repo        : $REPO_DIR"
echo "   environment : $ENV_NAME"
echo "=============================================="

# --- Step 1: find conda, or install Miniconda --------------------------------
echo
echo "[1/4] Locating conda"

if [ -n "$CONDA_MODULE" ] && command -v module &> /dev/null; then
    echo "  loading module $CONDA_MODULE"
    module load "$CONDA_MODULE" 2>/dev/null || echo "  module load failed, continuing"
fi

# An installation this script made earlier will not be on PATH in a new shell.
if ! command -v conda &> /dev/null && [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi

if ! command -v conda &> /dev/null; then
    echo "  conda not found, installing Miniconda into $CONDA_ROOT"

    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)   MINI="Miniconda3-latest-Linux-x86_64.sh" ;;
        Linux-aarch64)  MINI="Miniconda3-latest-Linux-aarch64.sh" ;;
        Darwin-arm64)   MINI="Miniconda3-latest-MacOSX-arm64.sh" ;;
        Darwin-x86_64)  MINI="Miniconda3-latest-MacOSX-x86_64.sh" ;;
        *)
            echo "ERROR: unsupported platform $(uname -s)-$(uname -m)."
            echo "  Install Miniconda manually from https://docs.conda.io/en/latest/miniconda.html"
            echo "  then re-run this script."
            exit 1 ;;
    esac

    INSTALLER="/tmp/$MINI"
    echo "  downloading $MINI"
    if command -v curl &> /dev/null; then
        curl -fsSL "https://repo.anaconda.com/miniconda/$MINI" -o "$INSTALLER"
    elif command -v wget &> /dev/null; then
        wget -q "https://repo.anaconda.com/miniconda/$MINI" -O "$INSTALLER"
    else
        echo "ERROR: neither curl nor wget is available to download Miniconda."
        exit 1
    fi

    # -b batch mode, -p prefix. No shell profile is modified.
    bash "$INSTALLER" -b -p "$CONDA_ROOT"
    rm -f "$INSTALLER"

    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    echo "  Miniconda installed"
fi

echo "  conda: $(command -v conda)"

# Make "conda activate" work inside this non-interactive shell.
eval "$(conda shell.bash hook)"

# Conda 25 and newer refuse to solve anything until the Terms of Service for
# Anaconda's default channels have been accepted, and they fail with a hard
# error in a non-interactive shell. That happens even though this project only
# installs from conda-forge, because those channels stay in the default
# configuration. Accept them up front so a fresh install does not stop here.
# The command does not exist on older conda, hence the guard.
if conda tos --help &> /dev/null; then
    for tos_channel in https://repo.anaconda.com/pkgs/main \
                       https://repo.anaconda.com/pkgs/r; do
        conda tos accept --override-channels --channel "$tos_channel" &> /dev/null || true
    done
    echo "  channel terms of service accepted"
fi

# --- Step 2: create the environment ------------------------------------------
echo
echo "[2/4] Creating environment"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  $ENV_NAME already exists, updating it to match environment.yml"
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "  building $ENV_NAME from environment.yml (this takes a few minutes)"
    conda env create -n "$ENV_NAME" -f environment.yml
fi

conda activate "$ENV_NAME"
echo "  python: $(command -v python) ($(python --version 2>&1))"

# --- Step 3: verify ----------------------------------------------------------
echo
echo "[3/4] Verifying imports"

python - <<'PYCHECK'
import importlib
import sys

modules = [
    ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
    ("matplotlib", "matplotlib"), ("statsmodels", "statsmodels"),
    ("Bio", "biopython"), ("requests", "requests"),
    ("tmtools", "tmtools"), ("rcsbapi", "rcsb-api"),
    ("metapredict", "metapredict"),
]

failed = []
for import_name, pip_name in modules:
    try:
        module = importlib.import_module(import_name)
        print("  {0:<14} {1}".format(pip_name, getattr(module, "__version__", "ok")))
    except Exception as exc:
        failed.append(pip_name)
        print("  {0:<14} FAILED: {1}".format(pip_name, exc))

if failed:
    print("\nMissing: " + ", ".join(failed))
    print("Try: conda env update -n $ENV_NAME -f environment.yml --prune")
    sys.exit(1)
print("\nAll imports OK.")
PYCHECK

# --- Step 4: optional pre-download -------------------------------------------
echo
echo "[4/4] Structure cache"

if [ "$DO_DOWNLOAD" = "1" ]; then
    if [ ! -f proteins.csv ]; then
        echo "  proteins.csv not found, building the registry first"
        python src/build_registry.py
    fi
    echo "  downloading structures into ./data (this takes a while)"
    python src/af_study.py
    echo "  cache ready: $(ls data 2>/dev/null | wc -l) files in ./data"
else
    echo "  skipped. Re-run with --download to populate ./data before submitting jobs."
fi

# --- Done --------------------------------------------------------------------
cat <<EOF

==============================================
 Done.

 Every new shell needs these two lines:

   source $CONDA_ROOT/etc/profile.d/conda.sh
   conda activate $ENV_NAME

 Then run the study:

   python src/af_study.py

 For cluster jobs, put the same two lines into each jobs/batch_NN.sh,
 and fill in the account and partition at the top of those files.
==============================================
EOF

#!/bin/bash
# =============================================================================
# setup_env.sh
#
# Creates the conda environment this project needs, on a login node.
#
# Run this ONCE before submitting any jobs. Compute nodes on most clusters have
# no outbound network access, so both the package install and the structure
# downloads have to happen here, on a node that does. The batch jobs then only
# read from shared storage.
#
#   bash setup_env.sh              # create the environment
#   bash setup_env.sh --download   # also pre-download every structure
#
# -----------------------------------------------------------------------------
# EDIT THESE FOR YOUR CLUSTER BEFORE RUNNING
# -----------------------------------------------------------------------------

# Name of the environment to create.
ENV_NAME="alphafold-study"

# Python version. 3.11 is known to work with every dependency below.
PY_VERSION="3.11"

# Where the environment is written. On most clusters your home directory has a
# small quota and conda environments are large (torch alone is over 1 GB), so
# point this at scratch or project space rather than $HOME.
ENV_PREFIX="${HOME}/.conda/envs/${ENV_NAME}"

# Where package downloads are cached. Same quota warning applies.
export CONDA_PKGS_DIRS="${HOME}/.conda/pkgs"

# Where pip caches wheels.
export PIP_CACHE_DIR="${HOME}/.cache/pip"

# Module providing conda. Common values: anaconda3, miniconda3, conda.
# Run "module avail conda" to find the right name, or leave empty if conda is
# already on your PATH.
CONDA_MODULE="anaconda3"

# =============================================================================
# Nothing below here normally needs editing.
# =============================================================================

set -euo pipefail

echo "=============================================="
echo " Environment : ${ENV_NAME}"
echo " Python      : ${PY_VERSION}"
echo " Prefix      : ${ENV_PREFIX}"
echo "=============================================="

# --- Locate conda ------------------------------------------------------------
if [ -n "${CONDA_MODULE}" ] && command -v module &> /dev/null; then
    echo "[1/4] Loading module ${CONDA_MODULE}"
    module load "${CONDA_MODULE}" || echo "  could not load ${CONDA_MODULE}, trying PATH"
else
    echo "[1/4] Skipping module load, using conda from PATH"
fi

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found."
    echo "  Set CONDA_MODULE at the top of this script, or load conda yourself first."
    echo "  Try: module avail conda"
    exit 1
fi
echo "  conda: $(command -v conda)"

# Make "conda activate" work inside a non-interactive shell.
eval "$(conda shell.bash hook)"

# --- Create the environment --------------------------------------------------
if conda env list | grep -qE "^${ENV_NAME}\s|/${ENV_NAME}$"; then
    echo "[2/4] Environment ${ENV_NAME} already exists, reusing it"
else
    echo "[2/4] Creating ${ENV_NAME} with Python ${PY_VERSION}"
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"
echo "  python: $(command -v python) ($(python --version 2>&1))"

# --- Install dependencies ----------------------------------------------------
# Installed with pip rather than conda so the versions match what the analysis
# was developed against. Torch is pinned to the CPU build: metapredict needs
# torch, nothing here uses a GPU, and the CUDA build is several GB larger.
echo "[3/4] Installing dependencies"

python -m pip install --upgrade pip --quiet

echo "  torch (CPU build)"
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

echo "  scientific stack and project dependencies"
python -m pip install --quiet \
    numpy \
    scipy \
    pandas \
    matplotlib \
    statsmodels \
    biopython \
    tmtools \
    requests \
    rcsb-api \
    metapredict

# --- Verify ------------------------------------------------------------------
echo "[4/4] Verifying imports"
python - <<'PYCHECK'
import importlib
import sys

modules = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("statsmodels", "statsmodels"),
    ("Bio", "biopython"),
    ("tmtools", "tmtools"),
    ("requests", "requests"),
    ("rcsbapi", "rcsb-api"),
    ("metapredict", "metapredict"),
]

failed = []
for import_name, pip_name in modules:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "ok")
        print("  {0:<14} {1}".format(pip_name, version))
    except Exception as exc:
        failed.append(pip_name)
        print("  {0:<14} FAILED: {1}".format(pip_name, exc))

if failed:
    print("\nMissing: " + ", ".join(failed))
    sys.exit(1)
print("\nAll imports OK.")
PYCHECK

# --- Optional pre-download ---------------------------------------------------
# Compute nodes cannot reach RCSB or the AlphaFold database, so every structure
# has to be cached on shared storage first. af_study.py reuses whatever is
# already in ./data, so running it here populates the cache; the batch jobs then
# find every file present and skip straight to the analysis.
if [ "${1:-}" = "--download" ]; then
    echo
    echo "Pre-downloading structures into ./data (this takes a while)"
    python src/af_study.py
    echo "Download cache ready: $(ls data 2>/dev/null | wc -l) files in ./data"
fi

echo
echo "=============================================="
echo " Done."
echo
echo " Activate it with:"
echo "   conda activate ${ENV_NAME}"
echo
echo " Then put these two lines in each jobs/batch_NN.sh:"
echo "   module load ${CONDA_MODULE}"
echo "   conda activate ${ENV_NAME}"
echo
echo " If you have not pre-downloaded yet, run:"
echo "   bash setup_env.sh --download"
echo "=============================================="

#!/usr/bin/env python3
"""af_study.py

Compare AlphaFold2 prediction accuracy on viral versus human cellular
proteins by measuring AlphaFold models against experimental PDB structures.

The whole study runs top to bottom in this one file:

  Stage 1  Download experimental (RCSB) and AlphaFold structures into ./data
  Stage 2  Compute per-protein metrics into ./results/metrics.csv
  Stage 3  Run exploratory statistics into ./results/stats.txt
  Stage 4  Render three figures into ./results/

The protein registry comes from proteins.csv when present (see
build_registry.py); otherwise a small built-in list is used so the script
works out of the box. Adding proteins only requires editing proteins.csv (or
the built-in list) and re-running.

The script is idempotent: existing downloads are reused, and results are
overwritten on every run.

Run with:  python af_study.py
"""

import csv
import importlib
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Dependency bootstrap: install on first run, fall back to system override.
# ---------------------------------------------------------------------------
def ensure_deps():
    """Install required third-party packages if they are missing.

    Core packages are mandatory. metapredict is optional: if it cannot be
    installed or imported the disorder metric degrades to "NA" rather than
    failing the run.
    """
    core = {
        "requests": "requests",
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "Bio": "biopython",
        "tmtools": "tmtools",
    }
    optional = {
        "metapredict": "metapredict",
    }

    def pip_install(pip_names):
        base = [sys.executable, "-m", "pip", "install", "--quiet"]
        try:
            subprocess.check_call(base + pip_names)
        except subprocess.CalledProcessError:
            print("Standard install failed, retrying with --break-system-packages")
            subprocess.check_call(base + ["--break-system-packages"] + pip_names)

    missing_core = {}
    for import_name, pip_name in core.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_core[import_name] = pip_name
    if missing_core:
        pip_names = sorted(set(missing_core.values()))
        print("Installing core dependencies: " + ", ".join(pip_names))
        pip_install(pip_names)
        importlib.invalidate_caches()

    for import_name, pip_name in optional.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            print("Installing optional dependency: " + pip_name)
            try:
                pip_install([pip_name])
                importlib.invalidate_caches()
            except Exception as exc:
                print("Optional dependency {0} unavailable: {1}".format(pip_name, exc))


ensure_deps()

import numpy as np  # noqa: E402
import requests  # noqa: E402
from Bio.PDB import MMCIFParser  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr, kruskal  # noqa: E402
from tmtools import tm_align  # noqa: E402
from tmtools.io import get_structure, get_residue_data  # noqa: E402

# metapredict is optional. Probe it once and degrade gracefully.
try:
    import metapredict as _metapredict  # noqa: E402
    METAPREDICT_OK = True
except Exception as _exc:  # pragma: no cover - environment dependent
    _metapredict = None
    METAPREDICT_OK = False
    print("metapredict not available, disorder will be reported as NA: {0}".format(_exc))


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir": "./data",
    "results_dir": "./results",
    "registry_csv": "proteins.csv",

    # AlphaFold model file versions to try, in order. v4 is the version named
    # in the study brief; the AlphaFold DB has since moved on, so newer
    # versions are listed as fall-backs so downloads still work for real use.
    "af_model_versions": ["v4", "v6", "v5", "v3"],

    # When the classic file URL does not resolve, fall back to the AlphaFold
    # API to find the real model file. The AlphaFold DB has migrated many
    # entries (notably viral ones) to a hash-based ID scheme that the classic
    # UniProt-accession URL no longer serves. For multi-fragment proteins the
    # first model returned by the API is used. Set to False for classic-only.
    "use_af_api_fallback": True,

    "rcsb_url": "https://files.rcsb.org/download/{pdb}.pdb",
    # Many modern or large entries have no legacy PDB-format file and are only
    # served as mmCIF, so the experimental download falls back to .cif.
    "rcsb_cif_url": "https://files.rcsb.org/download/{pdb}.cif",
    "af_url": "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_{version}.pdb",
    "af_api_url": "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}",

    "request_sleep": 0.3,
    "request_timeout": 60,

    # A residue counts as disordered when its metapredict score is at or above
    # this threshold.
    "disorder_threshold": 0.5,

    "figure_dpi": 150,
}


# Built-in fallback registry, used only when proteins.csv is absent.
# Columns: name, type, pdb_id, pdb_chain, uniprot.
#
# Caveat: several viral entries here map a mature-protein crystal structure to a
# polyprotein UniProt accession (for example 6LU7 main protease -> P0DTD1, the
# pp1ab polyprotein). AlphaFold serves such accessions as fragments, so the
# first fragment may not contain the domain in the PDB file, giving a low
# TM-score that reflects fragment mismatch rather than prediction error. This is
# a property of these hand-picked accessions, not of the pipeline. For a clean,
# balanced comparison generate the registry with build_registry.py, which keeps
# one single-domain representative per accession inside a length window.
BUILTIN_REGISTRY = [
    {"name": "SARS-CoV-2 Main Protease", "type": "viral", "pdb_id": "6LU7", "pdb_chain": "A", "uniprot": "P0DTD1"},
    {"name": "HIV-1 Protease", "type": "viral", "pdb_id": "1HSG", "pdb_chain": "A", "uniprot": "P04585"},
    {"name": "Influenza Neuraminidase", "type": "viral", "pdb_id": "2HU4", "pdb_chain": "A", "uniprot": "P03468"},
    {"name": "SARS-CoV-2 Spike RBD", "type": "viral", "pdb_id": "6M0J", "pdb_chain": "E", "uniprot": "P0DTC2"},
    {"name": "Dengue NS3 Helicase", "type": "viral", "pdb_id": "2BMF", "pdb_chain": "A", "uniprot": "Q9YID8"},
    {"name": "Human Lysozyme", "type": "cellular", "pdb_id": "1LZ1", "pdb_chain": "A", "uniprot": "P61626"},
    {"name": "Human Serum Albumin", "type": "cellular", "pdb_id": "1AO6", "pdb_chain": "A", "uniprot": "P02768"},
    {"name": "Human Ubiquitin", "type": "cellular", "pdb_id": "1UBQ", "pdb_chain": "A", "uniprot": "P0CG48"},
    {"name": "Human Hemoglobin Alpha", "type": "cellular", "pdb_id": "1HHO", "pdb_chain": "A", "uniprot": "P69905"},
    {"name": "Human Cyclophilin A", "type": "cellular", "pdb_id": "1CWA", "pdb_chain": "A", "uniprot": "P62937"},
]


# ---------------------------------------------------------------------------
# Registry loading.
# ---------------------------------------------------------------------------
def load_registry():
    """Load the protein registry from proteins.csv, or fall back to built-in."""
    path = CONFIG["registry_csv"]
    if os.path.exists(path):
        rows = []
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cleaned = {k: (v.strip() if isinstance(v, str) else v)
                           for k, v in row.items()}
                if cleaned.get("pdb_id") and cleaned.get("uniprot"):
                    rows.append(cleaned)
        if rows:
            print("Loaded {0} proteins from {1}.".format(len(rows), path))
            return rows
        print("{0} is present but empty, using built-in registry.".format(path))
    else:
        print("No {0} found, using built-in registry of {1} proteins.".format(
            path, len(BUILTIN_REGISTRY)))
    return [dict(r) for r in BUILTIN_REGISTRY]


# ---------------------------------------------------------------------------
# Stage 1: download.
# ---------------------------------------------------------------------------
def http_download(url, dest, timeout):
    """Download url to dest. Return True on success, False otherwise."""
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as exc:
        print("  download error {0}: {1}".format(url, exc))
        return False
    if resp.status_code != 200 or not resp.content:
        return False
    with open(dest, "wb") as handle:
        handle.write(resp.content)
    return True


def local_structure_path(prefix):
    """Return an existing local structure file (prefix + .pdb or .cif), or None."""
    for ext in (".pdb", ".cif"):
        candidate = prefix + ext
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def experimental_prefix(pdb_id):
    return os.path.join(CONFIG["data_dir"], pdb_id)


def alphafold_prefix(uniprot):
    return os.path.join(CONFIG["data_dir"], "AF-{0}".format(uniprot))


def download_experimental(pdb_id):
    """Download an experimental structure from RCSB, .pdb with a .cif fallback."""
    prefix = experimental_prefix(pdb_id)
    if local_structure_path(prefix):
        return "skip"
    attempts = [
        (".pdb", CONFIG["rcsb_url"].format(pdb=pdb_id)),
        (".cif", CONFIG["rcsb_cif_url"].format(pdb=pdb_id)),
    ]
    for ext, url in attempts:
        ok = http_download(url, prefix + ext, CONFIG["request_timeout"])
        time.sleep(CONFIG["request_sleep"])
        if ok:
            return "ok"
    return "fail"


def download_alphafold(uniprot):
    """Download an AlphaFold model. Return ok/skip/fail.

    Tries the classic UniProt-accession file URL across configured versions
    first, then falls back to the AlphaFold API to resolve the real file URL
    (the DB has migrated many entries to a hash-based ID scheme). The API path
    prefers the PDB file and falls back to mmCIF.
    """
    prefix = alphafold_prefix(uniprot)
    if local_structure_path(prefix):
        return "skip"
    for version in CONFIG["af_model_versions"]:
        url = CONFIG["af_url"].format(uniprot=uniprot, version=version)
        ok = http_download(url, prefix + ".pdb", CONFIG["request_timeout"])
        time.sleep(CONFIG["request_sleep"])
        if ok:
            return "ok"

    if CONFIG["use_af_api_fallback"]:
        api_url = CONFIG["af_api_url"].format(uniprot=uniprot)
        try:
            resp = requests.get(api_url, timeout=CONFIG["request_timeout"])
            if resp.status_code == 200:
                entries = resp.json()
                if entries:
                    for key, ext in [("pdbUrl", ".pdb"), ("cifUrl", ".cif")]:
                        file_url = entries[0].get(key)
                        if file_url and http_download(file_url, prefix + ext, CONFIG["request_timeout"]):
                            time.sleep(CONFIG["request_sleep"])
                            return "ok"
        except Exception as exc:
            print("  AlphaFold API fallback failed for {0}: {1}".format(uniprot, exc))
        time.sleep(CONFIG["request_sleep"])
    return "fail"


def stage1_download(registry):
    """Download every experimental and AlphaFold file needed by the study."""
    print("\n=== Stage 1: download structures into {0} ===".format(CONFIG["data_dir"]))
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    failures = []
    for protein in registry:
        pdb_id = protein["pdb_id"]
        uniprot = protein["uniprot"]

        exp_status = download_experimental(pdb_id)
        af_status = download_alphafold(uniprot)
        print("  {0:<32} exp({1})={2:<4} AF({3})={4}".format(
            protein["name"][:32], pdb_id, exp_status, uniprot, af_status))
        if exp_status == "fail":
            failures.append("{0} experimental {1}".format(protein["name"], pdb_id))
        if af_status == "fail":
            failures.append("{0} AlphaFold {1}".format(protein["name"], uniprot))

    if failures:
        print("  {0} download failure(s):".format(len(failures)))
        for item in failures:
            print("    - " + item)
    else:
        print("  All downloads present.")


# ---------------------------------------------------------------------------
# Stage 2: metrics.
# ---------------------------------------------------------------------------
def load_chain(path, chain_id):
    """Return (Bio chain, CA coords, sequence) for chain_id in a structure.

    Uses tmtools' loader for PDB files and Biopython's MMCIFParser for mmCIF.
    Both parsers expose author chain IDs, so chain_id matches the registry.
    Falls back to the first chain if the requested chain is not present.
    """
    if path.endswith(".cif") or path.endswith(".mmcif"):
        structure = MMCIFParser(QUIET=True).get_structure("structure", path)
    else:
        structure = get_structure(path)
    model = next(structure.get_models())
    if chain_id in model:
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())
    coords, seq = get_residue_data(chain)
    return chain, coords, seq


def mean_plddt(af_chain):
    """Mean CA B-factor of an AlphaFold chain, which encodes per-residue pLDDT."""
    values = [atom.get_bfactor() for residue in af_chain
              for atom in residue if atom.get_id() == "CA"]
    if not values:
        return None
    return float(np.mean(values))


def missing_fraction(exp_chain):
    """Fraction of residues missing across the observed numbering span.

    Used as a crystal-disorder proxy: residues that are part of the chain but
    absent from the coordinates leave gaps in the author residue numbering.
    """
    resnums = [residue.id[1] for residue in exp_chain if residue.has_id("CA")]
    if not resnums:
        return None
    lo, hi = min(resnums), max(resnums)
    span = hi - lo + 1
    if span <= 0:
        return None
    observed = len(set(resnums))
    return float((span - observed) / span)


def compute_disorder(sequence):
    """Fraction of residues predicted disordered by metapredict, or None."""
    if not METAPREDICT_OK or not sequence:
        return None
    try:
        scores = _metapredict.predict_disorder(sequence)
        if hasattr(scores, "disorder"):
            scores = scores.disorder
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return None
        return float(np.mean(scores >= CONFIG["disorder_threshold"]))
    except Exception as exc:
        print("  disorder prediction failed: {0}".format(exc))
        return None


def disorder_bin(disorder_pct):
    """Bucket a disorder percentage into 0-25 / 25-50 / 50-75 / 75-100."""
    if disorder_pct is None:
        return "NA"
    if disorder_pct < 25:
        return "0-25"
    if disorder_pct < 50:
        return "25-50"
    if disorder_pct < 75:
        return "50-75"
    return "75-100"


METRIC_FIELDS = [
    "name", "type", "pdb_id", "pdb_chain", "uniprot",
    "exp_len", "af_len", "tm_score", "rmsd", "mean_plddt",
    "disorder_frac", "disorder_pct", "disorder_bin", "missing_frac", "status",
]


def stage2_metrics(registry):
    """Compute metrics for every protein and write results/metrics.csv."""
    print("\n=== Stage 2: compute metrics ===")
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    rows = []
    for protein in registry:
        row = {
            "name": protein["name"],
            "type": protein["type"],
            "pdb_id": protein["pdb_id"],
            "pdb_chain": protein["pdb_chain"],
            "uniprot": protein["uniprot"],
            "exp_len": None, "af_len": None, "tm_score": None, "rmsd": None,
            "mean_plddt": None, "disorder_frac": None, "disorder_pct": None,
            "disorder_bin": "NA", "missing_frac": None, "status": "ok",
        }
        try:
            exp_path = local_structure_path(experimental_prefix(protein["pdb_id"]))
            af_path = local_structure_path(alphafold_prefix(protein["uniprot"]))
            if exp_path is None:
                raise FileNotFoundError("experimental file missing")
            if af_path is None:
                raise FileNotFoundError("AlphaFold file missing")

            exp_chain, exp_coords, exp_seq = load_chain(exp_path, protein["pdb_chain"])
            af_chain, af_coords, af_seq = load_chain(af_path, "A")
            row["exp_len"] = len(exp_seq)
            row["af_len"] = len(af_seq)
            if len(exp_seq) == 0 or len(af_seq) == 0:
                raise ValueError("empty chain after parsing")

            # chain1 is the experimental structure, so tm_norm_chain1 is the
            # TM-score normalised by the length of the true structure: how well
            # the AlphaFold model recovers the experimentally observed fold.
            result = tm_align(exp_coords, af_coords, exp_seq, af_seq)
            row["tm_score"] = float(result.tm_norm_chain1)
            row["rmsd"] = float(result.rmsd)

            row["mean_plddt"] = mean_plddt(af_chain)
            row["missing_frac"] = missing_fraction(exp_chain)

            disorder = compute_disorder(af_seq)
            if disorder is not None:
                row["disorder_frac"] = disorder
                row["disorder_pct"] = disorder * 100.0
                row["disorder_bin"] = disorder_bin(row["disorder_pct"])
        except Exception as exc:
            row["status"] = "error: {0}".format(exc)
            print("  {0}: {1}".format(protein["name"], row["status"]))

        if row["status"] == "ok":
            print("  {0:<32} TM={1} RMSD={2} pLDDT={3} disorder={4}".format(
                protein["name"][:32], _fmt(row["tm_score"], 3),
                _fmt(row["rmsd"], 2), _fmt(row["mean_plddt"], 1),
                _fmt(row["disorder_pct"], 1)))
        rows.append(row)

    out = os.path.join(CONFIG["results_dir"], "metrics.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row[k]) for k in METRIC_FIELDS})
    print("  Wrote metrics for {0} proteins to {1}.".format(len(rows), out))
    return rows


def _fmt(value, nd=4):
    """Format a value for CSV/printing, using NA for None."""
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "{0:.{1}f}".format(value, nd)
    return str(value)


# ---------------------------------------------------------------------------
# Stage 3: statistics.
# ---------------------------------------------------------------------------
def _median(values):
    return float(np.median(values)) if values else float("nan")


def rank_biserial(u_statistic, n1, n2):
    """Rank-biserial effect size from a Mann-Whitney U for the first sample.

    r = 2*U/(n1*n2) - 1, in [-1, 1]. Positive means the first group tends to
    have the higher values.
    """
    if n1 == 0 or n2 == 0:
        return float("nan")
    return 2.0 * u_statistic / (n1 * n2) - 1.0


def stage3_stats(rows):
    """Run exploratory statistics, print them, and write results/stats.txt."""
    print("\n=== Stage 3: statistics ===")
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    ok = [r for r in rows if r["status"] == "ok" and r["tm_score"] is not None]
    viral = [r for r in ok if r["type"] == "viral"]
    cellular = [r for r in ok if r["type"] == "cellular"]

    emit("AlphaFold accuracy: viral vs cellular proteins")
    emit("Exploratory analysis. Effect sizes are reported alongside p-values;")
    emit("p-values are descriptive given the small, convenience-based sample.")
    emit("")
    emit("Usable proteins: {0} total ({1} viral, {2} cellular)".format(
        len(ok), len(viral), len(cellular)))
    emit("")

    # Mann-Whitney U on TM-score and RMSD.
    for metric, label in [("tm_score", "TM-score"), ("rmsd", "RMSD")]:
        v = [r[metric] for r in viral if r[metric] is not None]
        c = [r[metric] for r in cellular if r[metric] is not None]
        emit("Mann-Whitney U on {0} (viral vs cellular)".format(label))
        if len(v) >= 1 and len(c) >= 1:
            emit("  median viral    = {0:.4f} (n={1})".format(_median(v), len(v)))
            emit("  median cellular = {0:.4f} (n={1})".format(_median(c), len(c)))
            if len(v) >= 2 and len(c) >= 2:
                u_stat, p_value = mannwhitneyu(v, c, alternative="two-sided")
                r_rb = rank_biserial(u_stat, len(v), len(c))
                emit("  U = {0:.1f}, p = {1:.4g}".format(u_stat, p_value))
                emit("  rank-biserial r = {0:+.3f} (positive: viral higher)".format(r_rb))
            else:
                emit("  Not enough data in both groups for the U test (need n>=2 each).")
        else:
            emit("  No data available.")
        emit("")

    # Spearman correlations.
    def spearman_block(label, xs, ys, hint=""):
        emit("Spearman: {0}".format(label))
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if len(pairs) >= 3:
            xv = [p[0] for p in pairs]
            yv = [p[1] for p in pairs]
            rho, p_value = spearmanr(xv, yv)
            emit("  rho = {0:+.3f}, p = {1:.4g} (n={2})".format(rho, p_value, len(pairs)))
        else:
            emit("  Insufficient paired data (n={0}, need >=3).{1}".format(
                len(pairs), (" " + hint) if hint else ""))
        emit("")

    spearman_block(
        "disorder% vs TM-score (overall)",
        [r["disorder_pct"] for r in ok],
        [r["tm_score"] for r in ok],
        hint="metapredict may be unavailable.")
    spearman_block(
        "pLDDT vs TM-score (overall)",
        [r["mean_plddt"] for r in ok],
        [r["tm_score"] for r in ok])
    spearman_block(
        "pLDDT vs TM-score (viral only)",
        [r["mean_plddt"] for r in viral],
        [r["tm_score"] for r in viral])
    spearman_block(
        "pLDDT vs TM-score (cellular only)",
        [r["mean_plddt"] for r in cellular],
        [r["tm_score"] for r in cellular])

    # Kruskal-Wallis across disorder bins.
    emit("Kruskal-Wallis: TM-score across disorder bins")
    bins = {}
    for r in ok:
        b = r["disorder_bin"]
        if b and b != "NA" and r["tm_score"] is not None:
            bins.setdefault(b, []).append(r["tm_score"])
    populated = {b: vals for b, vals in bins.items() if len(vals) >= 2}
    for b in ["0-25", "25-50", "50-75", "75-100"]:
        if b in bins:
            emit("  bin {0:<7} n={1}".format(b, len(bins[b])))
    if len(populated) >= 2:
        h_stat, p_value = kruskal(*populated.values())
        emit("  H = {0:.3f}, p = {1:.4g} across {2} bins".format(
            h_stat, p_value, len(populated)))
    else:
        emit("  Fewer than 2 disorder bins have >=2 members.")
        emit("  Expand the protein set toward 30-40 per group (run build_registry.py)")
        emit("  to populate the disorder bins before interpreting this test.")
    emit("")

    out = os.path.join(CONFIG["results_dir"], "stats.txt")
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("  Wrote statistics to {0}.".format(out))


# ---------------------------------------------------------------------------
# Stage 4: figures.
# ---------------------------------------------------------------------------
def stage4_figures(rows):
    """Render three PNG figures into the results directory."""
    print("\n=== Stage 4: figures ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    dpi = CONFIG["figure_dpi"]
    colors = {"viral": "#d1495b", "cellular": "#30638e"}

    ok = [r for r in rows if r["status"] == "ok" and r["tm_score"] is not None]
    viral = [r for r in ok if r["type"] == "viral"]
    cellular = [r for r in ok if r["type"] == "cellular"]

    # Figure 1: TM-score boxplot with jittered points.
    fig, ax = plt.subplots(figsize=(6, 5))
    groups = [("viral", viral), ("cellular", cellular)]
    box_data = [[r["tm_score"] for r in grp] for _, grp in groups]
    positions = [1, 2]
    if any(box_data):
        ax.boxplot(box_data, positions=positions, widths=0.5,
                   showfliers=False, medianprops={"color": "black"})
    rng = np.random.default_rng(0)
    for pos, (name, grp) in zip(positions, groups):
        ys = [r["tm_score"] for r in grp]
        xs = pos + (rng.random(len(ys)) - 0.5) * 0.2
        ax.scatter(xs, ys, color=colors[name], alpha=0.7,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(["viral", "cellular"])
    ax.set_ylabel("TM-score (AF vs experimental)")
    ax.set_title("AlphaFold accuracy: viral vs cellular")
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["results_dir"], "tm_boxplot.png"), dpi=dpi)
    plt.close(fig)

    # Figure 2: pLDDT vs TM-score scatter coloured by type.
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, grp in groups:
        xs = [r["mean_plddt"] for r in grp if r["mean_plddt"] is not None]
        ys = [r["tm_score"] for r in grp if r["mean_plddt"] is not None]
        ax.scatter(xs, ys, color=colors[name], alpha=0.8, label=name,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("mean pLDDT")
    ax.set_ylabel("TM-score")
    ax.set_title("pLDDT vs TM-score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["results_dir"], "plddt_vs_tm.png"), dpi=dpi)
    plt.close(fig)

    # Figure 3: disorder% vs TM-score scatter coloured by type.
    fig, ax = plt.subplots(figsize=(6, 5))
    any_disorder = False
    for name, grp in groups:
        xs = [r["disorder_pct"] for r in grp if r["disorder_pct"] is not None]
        ys = [r["tm_score"] for r in grp if r["disorder_pct"] is not None]
        if xs:
            any_disorder = True
        ax.scatter(xs, ys, color=colors[name], alpha=0.8, label=name,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("predicted disorder (%)")
    ax.set_ylabel("TM-score")
    ax.set_title("Disorder vs TM-score")
    ax.legend()
    if not any_disorder:
        ax.text(0.5, 0.5, "disorder unavailable (metapredict not installed)",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["results_dir"], "disorder_vs_tm.png"), dpi=dpi)
    plt.close(fig)

    print("  Wrote tm_boxplot.png, plddt_vs_tm.png, disorder_vs_tm.png to {0}.".format(
        CONFIG["results_dir"]))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    print("AlphaFold viral vs cellular accuracy study")
    registry = load_registry()
    stage1_download(registry)
    rows = stage2_metrics(registry)
    stage3_stats(rows)
    stage4_figures(rows)
    print("\nDone. See {0} for metrics.csv, stats.txt and figures.".format(
        CONFIG["results_dir"]))


if __name__ == "__main__":
    main()

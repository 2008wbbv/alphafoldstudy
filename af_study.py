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
        "statsmodels": "statsmodels",
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
from Bio.Align import PairwiseAligner  # noqa: E402
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

# A protein is treated as a fragment mismatch (not a real prediction failure)
# when less than this fraction of its experimental sequence is present in the
# AlphaFold model. Tune here.
COVERAGE_MIN = 0.80


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
def get_chain(path, chain_id):
    """Return the requested Bio chain from a structure file.

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
        return model[chain_id]
    return next(model.get_chains())


def load_chain(path, chain_id):
    """Return (Bio chain, CA coords, sequence) for chain_id in a structure."""
    chain = get_chain(path, chain_id)
    coords, seq = get_residue_data(chain)
    return chain, coords, seq


def residue_level_data(chain):
    """Return (coords, seq, resnums, bfactors) parallel across CA residues.

    Matches the residue set used by tmtools.get_residue_data (standard residues
    that carry a CA atom) so the CA coordinates, author residue numbers and CA
    B-factors all line up index for index.
    """
    coords, seq = get_residue_data(chain)
    resnums, bfactors = [], []
    for residue in chain.get_residues():
        if residue.id[0] == " " and "CA" in residue.child_dict:
            resnums.append(residue.id[1])
            bfactors.append(float(residue.child_dict["CA"].get_bfactor()))
    if len(resnums) != len(coords):
        raise ValueError("residue numbering and coordinate counts disagree")
    return coords, seq, resnums, bfactors


def per_residue_disorder(sequence):
    """Per-residue metapredict disorder scores as a numpy array, or None."""
    if not METAPREDICT_OK or not sequence:
        return None
    try:
        scores = _metapredict.predict_disorder(sequence)
        if hasattr(scores, "disorder"):
            scores = scores.disorder
        return np.asarray(scores, dtype=float)
    except Exception:
        return None


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


# Aligner used only to measure sequence coverage. Match-only scoring with free
# end gaps so a short crystallised domain can align inside a long AlphaFold
# model without paying for the overhang.
_COVERAGE_ALIGNER = PairwiseAligner()
_COVERAGE_ALIGNER.mode = "global"
_COVERAGE_ALIGNER.match_score = 1.0
_COVERAGE_ALIGNER.mismatch_score = 0.0
_COVERAGE_ALIGNER.open_gap_score = -10.0
_COVERAGE_ALIGNER.extend_gap_score = -0.5
_COVERAGE_ALIGNER.end_gap_score = 0.0


def sequence_coverage(exp_seq, af_seq):
    """Fraction of the experimental sequence present in the AlphaFold model.

    Globally aligns the experimental chain sequence to the AlphaFold model
    sequence and returns (residues aligned to an identical AlphaFold residue) /
    (length of the experimental sequence). Low coverage means the model (often a
    polyprotein fragment) does not contain the crystallised region, so a low
    TM-score is a fragment-mismatch artifact rather than a real prediction
    failure. Returns None if it cannot be computed.
    """
    if not exp_seq or not af_seq:
        return None
    try:
        alignment = _COVERAGE_ALIGNER.align(exp_seq, af_seq)[0]
        identical = alignment.counts().identities
        return identical / len(exp_seq)
    except Exception as exc:
        print("  coverage computation failed: {0}".format(exc))
        return None


METRIC_FIELDS = [
    "name", "type", "pdb_id", "pdb_chain", "uniprot",
    "exp_len", "af_len", "tm_score", "rmsd", "mean_plddt",
    "disorder_frac", "disorder_pct", "disorder_bin", "missing_frac",
    "coverage", "fragment_flag", "status",
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
            "disorder_bin": "NA", "missing_frac": None,
            "coverage": None, "fragment_flag": False, "status": "ok",
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

            coverage = sequence_coverage(exp_seq, af_seq)
            row["coverage"] = coverage
            row["fragment_flag"] = coverage is not None and coverage < COVERAGE_MIN

            disorder = compute_disorder(af_seq)
            if disorder is not None:
                row["disorder_frac"] = disorder
                row["disorder_pct"] = disorder * 100.0
                row["disorder_bin"] = disorder_bin(row["disorder_pct"])
        except Exception as exc:
            row["status"] = "error: {0}".format(exc)
            print("  {0}: {1}".format(protein["name"], row["status"]))

        if row["status"] == "ok":
            print("  {0:<32} TM={1} RMSD={2} pLDDT={3} disorder={4} cov={5}{6}".format(
                protein["name"][:32], _fmt(row["tm_score"], 3),
                _fmt(row["rmsd"], 2), _fmt(row["mean_plddt"], 1),
                _fmt(row["disorder_pct"], 1), _fmt(row["coverage"], 2),
                "  [FRAGMENT]" if row["fragment_flag"] else ""))
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
# Stage 2b: per-residue analysis.
# ---------------------------------------------------------------------------
PER_RESIDUE_FIELDS = [
    "name", "type", "residue_index", "ca_distance", "plddt", "disorder_score",
]


def _per_residue_rows(protein):
    """Return per-residue records for one ok protein, or raise on failure.

    Re-runs the TM-align superposition, walks the residue-level alignment it
    returns, and for every aligned experimental/AF residue pair records the CA
    distance after superposition, the AF pLDDT and the metapredict disorder
    score at that experimental position.
    """
    exp_path = local_structure_path(experimental_prefix(protein["pdb_id"]))
    af_path = local_structure_path(alphafold_prefix(protein["uniprot"]))
    if exp_path is None or af_path is None:
        raise FileNotFoundError("structure file missing")

    exp_chain = get_chain(exp_path, protein["pdb_chain"])
    af_chain = get_chain(af_path, "A")
    exp_coords, exp_seq, exp_resnums, _ = residue_level_data(exp_chain)
    af_coords, af_seq, _, af_bfactors = residue_level_data(af_chain)
    if len(exp_seq) == 0 or len(af_seq) == 0:
        raise ValueError("empty chain after parsing")

    result = tm_align(exp_coords, af_coords, exp_seq, af_seq)
    rot = np.asarray(result.u)
    trans = np.asarray(result.t)
    disorder = per_residue_disorder(exp_seq)

    records = []
    i = j = 0
    for res_x, res_y in zip(result.seqxA, result.seqyA):
        if res_x != "-" and res_y != "-":
            # Superimpose the experimental CA onto the AF frame (u @ x + t) and
            # measure the residual distance to its aligned AF CA.
            moved = rot @ exp_coords[i] + trans
            distance = float(np.linalg.norm(moved - af_coords[j]))
            dscore = None
            if disorder is not None and i < len(disorder):
                dscore = float(disorder[i])
            records.append({
                "name": protein["name"],
                "type": protein["type"],
                "residue_index": exp_resnums[i],
                "ca_distance": distance,
                "plddt": af_bfactors[j],
                "disorder_score": dscore,
            })
        if res_x != "-":
            i += 1
        if res_y != "-":
            j += 1
    return records


def _spearman_line(emit, label, xs, ys):
    """Emit a Spearman correlation for paired arrays, guarding small samples."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    emit("Spearman: {0}".format(label))
    if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
        rho, p_value = spearmanr(xs, ys)
        emit("  rho = {0:+.3f}, p = {1:.4g} (n={2} residues)".format(rho, p_value, len(xs)))
    else:
        emit("  Insufficient or constant data (n={0}).".format(len(xs)))
    emit("")


def stage_per_residue(rows):
    """Build the per-residue table, its statistics and a hexbin figure."""
    print("\n=== Stage 2b: per-residue analysis ===")
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    per_residue = []
    used, skipped = 0, 0
    for protein in rows:
        if protein["status"] != "ok":
            continue
        try:
            records = _per_residue_rows(protein)
            per_residue.extend(records)
            used += 1
        except Exception as exc:
            skipped += 1
            print("  per-residue skipped for {0}: {1}".format(protein["name"], exc))
    print("  Built {0} residues from {1} proteins ({2} skipped).".format(
        len(per_residue), used, skipped))

    out_csv = os.path.join(CONFIG["results_dir"], "per_residue.csv")
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_RESIDUE_FIELDS)
        writer.writeheader()
        for row in per_residue:
            writer.writerow({k: _fmt(row[k]) for k in PER_RESIDUE_FIELDS})
    print("  Wrote {0}.".format(out_csv))

    _per_residue_stats(per_residue)
    _per_residue_figure(per_residue)


def _per_residue_stats(per_residue):
    """Correlations and a disordered-vs-ordered comparison over all residues."""
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    dist = np.array([r["ca_distance"] for r in per_residue], dtype=float)
    plddt = np.array([r["plddt"] if r["plddt"] is not None else np.nan
                      for r in per_residue], dtype=float)
    disorder = np.array([r["disorder_score"] if r["disorder_score"] is not None else np.nan
                         for r in per_residue], dtype=float)
    types = np.array([r["type"] for r in per_residue])

    emit("Per-residue analysis of local AlphaFold error")
    emit("Local error is the CA-CA distance after TM-align superposition.")
    emit("Exploratory; residues within a protein are not independent.")
    emit("")
    emit("Total residues: {0}".format(len(per_residue)))
    emit("  with a disorder score: {0}".format(int(np.sum(~np.isnan(disorder)))))
    emit("")

    _spearman_line(emit, "disorder score vs CA distance (all residues)", disorder, dist)
    _spearman_line(emit, "pLDDT vs CA distance (all residues)", plddt, dist)
    for label in ["viral", "cellular"]:
        sel = types == label
        _spearman_line(emit, "disorder score vs CA distance ({0} only)".format(label),
                       disorder[sel], dist[sel])
        _spearman_line(emit, "pLDDT vs CA distance ({0} only)".format(label),
                       plddt[sel], dist[sel])

    emit("Disordered (score >= 0.5) vs ordered (< 0.5) residues, CA distance")
    valid = ~np.isnan(disorder)
    dis_dist = dist[valid & (disorder >= 0.5)]
    ord_dist = dist[valid & (disorder < 0.5)]
    if len(dis_dist) >= 1 and len(ord_dist) >= 1:
        emit("  median disordered = {0:.3f} A (n={1})".format(
            float(np.median(dis_dist)), len(dis_dist)))
        emit("  median ordered    = {0:.3f} A (n={1})".format(
            float(np.median(ord_dist)), len(ord_dist)))
        if len(dis_dist) >= 2 and len(ord_dist) >= 2:
            u_stat, p_value = mannwhitneyu(dis_dist, ord_dist, alternative="two-sided")
            r_rb = rank_biserial(u_stat, len(dis_dist), len(ord_dist))
            emit("  Mann-Whitney U = {0:.1f}, p = {1:.4g}".format(u_stat, p_value))
            emit("  rank-biserial r = {0:+.3f} (positive: disordered residues have".format(r_rb))
            emit("  larger local error)")
        else:
            emit("  Not enough residues in both groups for the U test.")
    else:
        emit("  No disorder scores available (metapredict may be unavailable).")
    emit("")

    out = os.path.join(CONFIG["results_dir"], "per_residue_stats.txt")
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("  Wrote {0}.".format(out))


def _per_residue_figure(per_residue):
    """Hexbin of per-residue CA distance vs per-residue disorder score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array([r["disorder_score"] for r in per_residue
                   if r["disorder_score"] is not None], dtype=float)
    ys = np.array([r["ca_distance"] for r in per_residue
                   if r["disorder_score"] is not None], dtype=float)
    out = os.path.join(CONFIG["results_dir"], "per_residue_disorder_hexbin.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(xs) >= 10:
        hb = ax.hexbin(xs, ys, gridsize=40, bins="log", cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="log10(residue count)")
    else:
        ax.text(0.5, 0.5, "not enough residues with disorder scores",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("per-residue disorder score")
    ax.set_ylabel("per-residue CA distance (A)")
    ax.set_title("Local AlphaFold error vs disorder (per residue)")
    fig.tight_layout()
    fig.savefig(out, dpi=CONFIG["figure_dpi"])
    plt.close(fig)
    print("  Wrote {0}.".format(out))


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


def _stats_block(ok, emit):
    """Emit the full set of statistics for one set of usable rows."""
    viral = [r for r in ok if r["type"] == "viral"]
    cellular = [r for r in ok if r["type"] == "cellular"]

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


def stage3_stats(rows):
    """Run the statistics on all proteins and on the coverage-filtered set.

    The coverage-filtered block excludes fragment-mismatch artifacts (a small
    crystallised domain compared against an AlphaFold model, often a polyprotein
    fragment, that does not contain it), which is what separates real AlphaFold
    prediction errors from a sequence mismatch.
    """
    print("\n=== Stage 3: statistics ===")
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    ok = [r for r in rows if r["status"] == "ok" and r["tm_score"] is not None]
    kept = [r for r in ok if not r["fragment_flag"]]
    dropped = [r for r in ok if r["fragment_flag"]]

    emit("AlphaFold accuracy: viral vs cellular proteins")
    emit("Exploratory analysis. Effect sizes are reported alongside p-values;")
    emit("p-values are descriptive given the small, convenience-based sample.")
    emit("")

    emit("=" * 60)
    emit("ALL PROTEINS")
    emit("=" * 60)
    _stats_block(ok, emit)

    emit("=" * 60)
    emit("COVERAGE-FILTERED (coverage >= {0:.2f})".format(COVERAGE_MIN))
    emit("=" * 60)
    _stats_block(kept, emit)

    dropped_viral = sum(1 for r in dropped if r["type"] == "viral")
    dropped_cellular = sum(1 for r in dropped if r["type"] == "cellular")
    emit("dropped {0} ({1} viral, {2} cellular) as fragment mismatches".format(
        len(dropped), dropped_viral, dropped_cellular))

    out = os.path.join(CONFIG["results_dir"], "stats.txt")
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("  Wrote statistics to {0}.".format(out))


# ---------------------------------------------------------------------------
# Stage 4: figures.
# ---------------------------------------------------------------------------
def stage4_figures(rows):
    """Render three PNG figures from the coverage-filtered set.

    The boxplot uses only kept proteins. The scatter plots also show the
    fragment-flagged proteins, drawn with an 'x' marker so the outliers stay
    visible but are clearly distinguishable from the kept set.
    """
    print("\n=== Stage 4: figures ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    dpi = CONFIG["figure_dpi"]
    colors = {"viral": "#d1495b", "cellular": "#30638e"}

    ok = [r for r in rows if r["status"] == "ok" and r["tm_score"] is not None]
    kept = [r for r in ok if not r["fragment_flag"]]
    flagged = [r for r in ok if r["fragment_flag"]]
    groups = [("viral", [r for r in kept if r["type"] == "viral"]),
              ("cellular", [r for r in kept if r["type"] == "cellular"])]

    def scatter_with_flags(ax, field, xlabel, title):
        """Scatter field vs TM-score: kept as circles, flagged as x markers."""
        for name, grp in groups:
            xs = [r[field] for r in grp if r[field] is not None]
            ys = [r["tm_score"] for r in grp if r[field] is not None]
            ax.scatter(xs, ys, color=colors[name], alpha=0.8, label=name,
                       edgecolor="white", linewidth=0.5)
        for r in flagged:
            if r[field] is not None:
                ax.scatter(r[field], r["tm_score"], color=colors[r["type"]],
                           marker="x", s=55, linewidth=1.6, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TM-score")
        ax.set_title(title + " (coverage-filtered)")
        handles = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=colors["viral"],
                   markeredgecolor="white", color="w", label="viral"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=colors["cellular"],
                   markeredgecolor="white", color="w", label="cellular"),
        ]
        if flagged:
            handles.append(Line2D([], [], marker="x", linestyle="none", color="gray",
                                  label="fragment-flagged"))
        ax.legend(handles=handles)

    # Figure 1: TM-score boxplot with jittered points (kept set only).
    fig, ax = plt.subplots(figsize=(6, 5))
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
    ax.set_title("AlphaFold accuracy: viral vs cellular (coverage-filtered)")
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["results_dir"], "tm_boxplot.png"), dpi=dpi)
    plt.close(fig)

    # Figure 2: pLDDT vs TM-score scatter.
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_with_flags(ax, "mean_plddt", "mean pLDDT", "pLDDT vs TM-score")
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["results_dir"], "plddt_vs_tm.png"), dpi=dpi)
    plt.close(fig)

    # Figure 3: disorder% vs TM-score scatter.
    fig, ax = plt.subplots(figsize=(6, 5))
    any_disorder = any(r["disorder_pct"] is not None for r in ok)
    scatter_with_flags(ax, "disorder_pct", "predicted disorder (%)", "Disorder vs TM-score")
    if not any_disorder:
        ax.text(0.5, 0.5, "disorder unavailable (metapredict not installed)",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["results_dir"], "disorder_vs_tm.png"), dpi=dpi)
    plt.close(fig)

    print("  Wrote tm_boxplot.png, plddt_vs_tm.png, disorder_vs_tm.png to {0}.".format(
        CONFIG["results_dir"]))


# ---------------------------------------------------------------------------
# Stage 5: multiple regression.
# ---------------------------------------------------------------------------
def stage5_regression(rows):
    """Test whether viral origin predicts accuracy once disorder and coverage
    are controlled for, on the coverage-filtered set.

    Fits tm_score ~ disorder_pred + coverage + is_viral with OLS. If is_viral
    loses significance while disorder_pred keeps it, disorder is doing the work
    and "viral" was largely a proxy.
    """
    print("\n=== Stage 5: multiple regression ===")
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from patsy import dmatrices
    except Exception as exc:
        print("  statsmodels unavailable, skipping regression: {0}".format(exc))
        return

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    formula = "tm_score ~ disorder_pred + coverage + is_viral"

    # Build the regression dataset from the coverage-filtered rows, dropping any
    # row missing one of the four modelled columns.
    records = []
    for r in rows:
        if r["status"] != "ok" or r["fragment_flag"]:
            continue
        tm = r["tm_score"]
        disorder_pred = r["disorder_frac"]
        coverage = r["coverage"]
        if tm is None or disorder_pred is None or coverage is None:
            continue
        records.append({
            "tm_score": tm,
            "disorder_pred": disorder_pred,
            "coverage": coverage,
            "is_viral": 1 if r["type"] == "viral" else 0,
        })

    n = len(records)
    emit("Multiple regression on the coverage-filtered set")
    emit("Model: {0}".format(formula))
    emit("Question: does viral origin still predict AlphaFold accuracy once")
    emit("disorder and coverage are controlled for?")
    emit("")
    emit("Sample size used (after dropping missing rows): n = {0}".format(n))
    if n < 20:
        emit("WARNING: n < 20, this regression is underpowered and should be")
        emit("treated as exploratory.")
    emit("")

    out = os.path.join(CONFIG["results_dir"], "regression.txt")
    if n < 5:
        emit("Not enough data to fit the model (need at least 5 rows).")
        with open(out, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print("  Wrote regression to {0}.".format(out))
        return

    predictors = ["disorder_pred", "coverage", "is_viral"]
    try:
        df = pd.DataFrame(records)
        full = smf.ols(formula, data=df).fit()
        disorder_only = smf.ols("tm_score ~ disorder_pred", data=df).fit()

        emit("Full model coefficients:")
        significance = {}
        for name in predictors:
            coef = full.params[name]
            p_value = full.pvalues[name]
            is_sig = p_value < 0.05
            significance[name] = is_sig
            emit("  {0:14} coef = {1:+.4f}, p = {2:.4g} -> {3}".format(
                name, coef, p_value,
                "significant at 0.05" if is_sig else "not significant at 0.05"))
        emit("")

        p_viral = full.pvalues["is_viral"]
        if significance["is_viral"]:
            emit("CORE ANSWER: is_viral REMAINS significant (p = {0:.4g}) after".format(p_viral))
            emit("controlling for disorder and coverage; viral origin carries")
            emit("predictive value of its own.")
        else:
            emit("CORE ANSWER: is_viral is NOT significant (p = {0:.4g}) after".format(p_viral))
            emit("controlling for disorder and coverage; the viral vs cellular gap")
            emit("is accounted for by disorder and coverage.")
        emit("")

        emit("Variance explained (R-squared):")
        emit("  disorder-only model  = {0:.4f}".format(disorder_only.rsquared))
        emit("  full model           = {0:.4f}".format(full.rsquared))
        emit("")

        emit("Variance inflation factors (multicollinearity, flag if > 5):")
        _, design = dmatrices(formula, data=df, return_type="dataframe")
        for i, name in enumerate(design.columns):
            if name == "Intercept":
                continue
            vif = variance_inflation_factor(design.values, i)
            flag = "  <-- HIGH (> 5)" if vif > 5 else ""
            emit("  {0:14} VIF = {1:.3f}{2}".format(name, vif, flag))
        emit("")

        with open(out, "w") as handle:
            handle.write(str(full.summary()))
            handle.write("\n\n")
            handle.write("\n".join(lines))
            handle.write("\n")
    except Exception as exc:
        emit("Regression failed: {0}".format(exc))
        with open(out, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    print("  Wrote regression to {0}.".format(out))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(registry_csv=None, results_dir=None):
    # Optional overrides let one invocation process a batch registry into its own
    # results subfolder (used by the HPC batch scripts). With no arguments the
    # behavior is unchanged: proteins.csv into ./results.
    if registry_csv:
        CONFIG["registry_csv"] = registry_csv
    if results_dir:
        CONFIG["results_dir"] = results_dir

    print("AlphaFold viral vs cellular accuracy study")
    print("Registry: {0} | results: {1}".format(
        CONFIG["registry_csv"], CONFIG["results_dir"]))
    registry = load_registry()
    stage1_download(registry)
    rows = stage2_metrics(registry)
    stage_per_residue(rows)
    stage3_stats(rows)
    stage4_figures(rows)
    stage5_regression(rows)
    print("\nDone. See {0} for metrics.csv, per_residue.csv, stats.txt,".format(
        CONFIG["results_dir"]))
    print("regression.txt and figures.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the AlphaFold viral vs cellular accuracy study.")
    parser.add_argument(
        "--registry", default=None,
        help="registry CSV to read (default: proteins.csv, or the built-in list)")
    parser.add_argument(
        "--results-dir", default=None,
        help="directory for outputs (default: ./results)")
    args = parser.parse_args()
    main(args.registry, args.results_dir)

#!/usr/bin/env python3
"""build_registry.py

Systematically select a balanced set of viral and human cellular proteins
from the RCSB Protein Data Bank and write them to proteins.csv for use by
af_study.py.

The point of this script is that the protein IDs come from the database
itself rather than from memory. That avoids typos in PDB or UniProt
accessions and avoids the selection bias that creeps in when a human picks
"famous" structures by hand.

Pipeline per group (viral, cellular):
  1. Run an RCSB structured search (taxonomy + protein + resolution).
  2. For each returned polymer-entity ID (for example "6LU7_1"), resolve
     details, including the sequence, through the REST data API.
  3. Keep one representative per UniProt accession, inside a length window.
  4. Confirm an AlphaFold model exists before accepting the candidate.

Then, across both groups, estimate disorder for every candidate with
metapredict and select proteins to deliberately populate all four disorder
bins as evenly as the available data allows, keeping the viral/cellular
balance within each bin as close to even as possible. A balance table is
printed so the make-up of the final set is visible.

All tunables live in the CONFIG dict at the top of the file.

Run with:  python build_registry.py
"""

import csv
import importlib
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Dependency bootstrap: install on first run, fall back to system override.
# ---------------------------------------------------------------------------
def ensure_deps():
    """Install third-party packages if missing.

    Core packages are mandatory. metapredict is optional: without it the
    disorder-balanced selection degrades to a simple first-N selection rather
    than failing the run.
    """
    core = {
        "requests": "requests",
        "rcsbapi": "rcsb-api",
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

    missing = {}
    for import_name, pip_name in core.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing[import_name] = pip_name
    if missing:
        pip_names = sorted(set(missing.values()))
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

import requests  # noqa: E402
from rcsbapi.search import AttributeQuery  # noqa: E402

try:
    import metapredict as _metapredict  # noqa: E402
    METAPREDICT_OK = True
except Exception as _exc:  # pragma: no cover - environment dependent
    _metapredict = None
    METAPREDICT_OK = False
    print("metapredict unavailable, disorder-balanced selection will fall back "
          "to first-N: {0}".format(_exc))


# ---------------------------------------------------------------------------
# Configuration. Edit these values to change the make-up of the registry.
# ---------------------------------------------------------------------------
CONFIG = {
    # Taxonomy lineage IDs used by the RCSB search.
    #   10239 = Viruses, 9606 = Homo sapiens.
    # The "cellular" label is used for the human group so that it matches the
    # vocabulary used downstream in af_study.py (viral vs cellular).
    "taxonomy": {"viral": "10239", "cellular": "9606"},

    # Structure-level filters.
    "polymer_type": "Protein",
    "max_resolution": 3.0,

    # Length window in residues for the selected entities.
    "min_length": 50,
    "max_length": 600,

    # Target size of the whole registry. The per-group and pool sizes are
    # raised in step with it: per_group_cap bounds how many viable candidates
    # are gathered per group (which bounds the disorder predictions), and
    # pool_size bounds how many raw search hits are examined per group. Distinct
    # viral proteins (and disordered crystallised proteins of either type) are
    # sparse, so the high-disorder bins usually fill less than the target; raise
    # pool_size and per_group_cap to push further into the search if needed.
    "target_total": 250,
    "per_group_cap": 125,
    "pool_size": 4000,

    # Fragment exclusion, applied at selection time.
    #
    # The AlphaFold database serves a protein as a single model only while its
    # UniProt sequence stays at or below this length; longer entries are split
    # into overlapping fragments (F1, F2, ...) and the pipeline downloads F1.
    # A crystal structure of a mature protein cleaved from a long polyprotein
    # therefore gets compared against an F1 fragment that need not contain it,
    # producing a near-zero TM-score that reflects a database indexing mismatch
    # rather than a prediction failure. Viral polyproteins are the main source.
    # Excluding long parent sequences here keeps those cases out of the dataset
    # entirely, instead of collecting them and filtering them afterwards.
    "af_max_single_fragment": 2700,
    "uniprot_api_url": "https://rest.uniprot.org/uniprotkb/{accession}.json?fields=length",

    # Candidate gathering checkpoints here after every accepted candidate, so a
    # long build that is interrupted resumes instead of starting over. Delete
    # this file to force a clean rebuild.
    "checkpoint_json": ".registry_checkpoint.json",

    # A residue counts as disordered when its metapredict score is at or above
    # this threshold; the disorder fraction is binned into four quartile bins.
    "disorder_threshold": 0.5,

    # AlphaFold model file versions to probe, in order. v4 is the version named
    # in the original study brief; the AlphaFold DB has since advanced, so the
    # newer versions are listed as fall-backs to keep the existence check (and
    # the matching download in af_study.py) working for real use.
    "af_model_versions": ["v4", "v6", "v5", "v3"],

    # When the classic file URL does not resolve, fall back to the AlphaFold
    # API. The AlphaFold DB has migrated many entries (notably viral ones) to a
    # hash-based ID scheme that the UniProt-accession file URL no longer serves,
    # so the API is needed to confirm those models exist for real use. Set to
    # False to restrict the existence check to the classic file URL only.
    "use_af_api_fallback": True,

    # Endpoints.
    "data_api_url": "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}",
    "entry_api_url": "https://data.rcsb.org/rest/v1/core/entry/{pdb}",
    "af_url": "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_{version}.pdb",
    "af_api_url": "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}",

    # Politeness and robustness.
    "request_sleep": 0.1,
    "request_timeout": 30,

    "output_csv": "proteins.csv",
}

DISORDER_BINS = ["0-25", "25-50", "50-75", "75-100"]
GROUP_TYPES = ["viral", "cellular"]


# ---------------------------------------------------------------------------
# RCSB search.
# ---------------------------------------------------------------------------
def build_query(taxonomy_id):
    """Build the combined attribute query for one taxonomy group."""
    q_tax = AttributeQuery(
        "rcsb_entity_source_organism.taxonomy_lineage.id",
        "exact_match",
        taxonomy_id,
    )
    q_type = AttributeQuery(
        "entity_poly.rcsb_entity_polymer_type",
        "exact_match",
        CONFIG["polymer_type"],
    )
    q_res = AttributeQuery(
        "rcsb_entry_info.resolution_combined",
        "less_or_equal",
        CONFIG["max_resolution"],
    )
    return q_tax & q_type & q_res


# ---------------------------------------------------------------------------
# REST data API: resolve one polymer entity to the fields we care about.
# ---------------------------------------------------------------------------
def resolve_entity(entity_id):
    """Resolve an entity ID such as "6LU7_1" to a detail dict, or None.

    Returns a dict with keys: pdb_id, pdb_chain, uniprot, name, length,
    sequence. Returns None when the entity has no UniProt mapping, no sequence,
    or the lookup fails.
    """
    if "_" not in entity_id:
        return None
    pdb, entity = entity_id.split("_", 1)
    url = CONFIG["data_api_url"].format(pdb=pdb, entity=entity)
    try:
        resp = requests.get(url, timeout=CONFIG["request_timeout"])
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print("  REST lookup failed for {0}: {1}".format(entity_id, exc))
        return None

    ident = data.get("rcsb_polymer_entity_container_identifiers", {}) or {}
    uniprot_ids = ident.get("uniprot_ids") or []
    auth_asym_ids = ident.get("auth_asym_ids") or []
    if not uniprot_ids or not auth_asym_ids:
        return None

    entity_poly = data.get("entity_poly", {}) or {}
    poly_entity = data.get("rcsb_polymer_entity", {}) or {}
    sequence = (entity_poly.get("pdbx_seq_one_letter_code_can")
                or entity_poly.get("pdbx_seq_one_letter_code") or "")

    return {
        "pdb_id": pdb.upper(),
        "pdb_chain": auth_asym_ids[0],
        "uniprot": uniprot_ids[0],
        "name": poly_entity.get("pdbx_description") or "Unknown",
        "length": entity_poly.get("rcsb_sample_sequence_length"),
        "sequence": sequence.strip().upper(),
    }


# ---------------------------------------------------------------------------
# AlphaFold model existence check.
# ---------------------------------------------------------------------------
def af_model_exists(uniprot):
    """Return a tag for the AlphaFold model source if one exists, else None.

    The API is asked first because it settles the question in one request. The
    versioned file URLs are the fall-back: probing those first costs up to four
    requests per accession and nearly always fails for the viral entries, whose
    models have moved to a hash-based naming scheme the old URLs never serve.
    That ordering dominated the runtime of a large build.
    """
    if CONFIG["use_af_api_fallback"]:
        url = CONFIG["af_api_url"].format(uniprot=uniprot)
        try:
            resp = requests.get(url, timeout=CONFIG["request_timeout"])
            if resp.status_code == 200 and resp.json():
                return "api"
        except Exception:
            pass

    for version in CONFIG["af_model_versions"]:
        url = CONFIG["af_url"].format(uniprot=uniprot, version=version)
        try:
            resp = requests.head(
                url, timeout=CONFIG["request_timeout"], allow_redirects=True
            )
            if resp.status_code == 200:
                return version
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Gather viable candidates for one group.
# ---------------------------------------------------------------------------
def fetch_resolution(pdb_id):
    """Best-effort resolution in Angstrom for a PDB entry, or None.

    Read from the entry-level REST record so the value can be written into the
    registry, which lets make_batches.py filter on resolution without needing
    the downloaded structure files.
    """
    url = CONFIG["entry_api_url"].format(pdb=pdb_id)
    try:
        resp = requests.get(url, timeout=CONFIG["request_timeout"])
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    values = (data.get("rcsb_entry_info", {}) or {}).get("resolution_combined") or []
    if values:
        try:
            return float(values[0])
        except (TypeError, ValueError):
            return None
    return None


def fetch_uniprot_length(accession):
    """Length of the full UniProt sequence for an accession, or None."""
    url = CONFIG["uniprot_api_url"].format(accession=accession)
    try:
        resp = requests.get(url, timeout=CONFIG["request_timeout"])
        resp.raise_for_status()
        return int(resp.json().get("sequence", {}).get("length"))
    except Exception:
        return None


def is_fragment_risk(accession):
    """True when AlphaFold serves this accession in fragments.

    A parent sequence longer than the single-model limit is split by the
    AlphaFold database, so the fragment the pipeline downloads may not contain
    the crystallised region at all. Those entries are excluded up front. An
    accession whose length cannot be determined is kept, because the runtime
    coverage check in af_study.py still guards against a mismatch.
    """
    length = fetch_uniprot_length(accession)
    if length is None:
        return False
    return length > CONFIG["af_max_single_fragment"]


def load_checkpoint():
    """Return (candidates, examined_entity_ids) from a previous partial run."""
    path = CONFIG["checkpoint_json"]
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path) as handle:
            data = json.load(handle)
        return data.get("candidates", []), set(data.get("examined", []))
    except Exception as exc:
        print("  checkpoint unreadable, starting fresh: {0}".format(exc))
        return [], set()


def save_checkpoint(candidates, examined):
    """Persist gathering progress so an interrupted build can resume."""
    tmp = CONFIG["checkpoint_json"] + ".tmp"
    try:
        with open(tmp, "w") as handle:
            json.dump({"candidates": candidates, "examined": sorted(examined)}, handle)
        os.replace(tmp, CONFIG["checkpoint_json"])
    except Exception as exc:
        print("  could not write checkpoint: {0}".format(exc))


def gather_candidates(type_label, taxonomy_id, prior=None, examined_ids=None):
    """Gather viable candidates for one group (no disorder binning yet).

    A candidate is viable when it passes every filter: one representative per
    UniProt accession, inside the length window, not served by AlphaFold as a
    fragment, and with an AlphaFold model that exists. Gathering resumes from
    any prior progress and stops at per_group_cap viable candidates or
    pool_size newly examined entities. Returns whatever was gathered even if
    the search raises partway through.
    """
    print("\n=== Gathering {0} candidates (taxonomy {1}) ===".format(
        type_label, taxonomy_id))
    candidates = [c for c in (prior or []) if c["type"] == type_label]
    seen_examined = examined_ids if examined_ids is not None else set()
    # Accessions already accepted in ANY group, so a resumed run does not
    # duplicate work or re-add a protein the checkpoint already holds.
    seen_uniprot = {c["uniprot"] for c in (prior or [])}
    if candidates:
        print("  resuming with {0} {1} candidate(s) from checkpoint.".format(
            len(candidates), type_label))
    fragments_skipped = 0

    try:
        query = build_query(taxonomy_id)
        results = query(return_type="polymer_entity")

        examined = 0
        for entity_id in results:
            if len(candidates) >= CONFIG["per_group_cap"]:
                break
            if examined >= CONFIG["pool_size"]:
                break
            # Entities looked at on an earlier run cost nothing to skip.
            if entity_id in seen_examined:
                continue
            examined += 1
            seen_examined.add(entity_id)
            # Checkpoint the examined set periodically as well as on every
            # acceptance. Viable candidates can be hundreds of entities apart,
            # and without this an interrupted run re-examines everything it
            # already rejected.
            if examined % 25 == 0:
                save_checkpoint(
                    [c for c in (prior or []) if c["type"] != type_label] + candidates,
                    seen_examined)
                print("    ... {0} examined, {1} {2} so far ({3} fragments skipped)".format(
                    len(seen_examined), len(candidates), type_label, fragments_skipped))

            details = resolve_entity(entity_id)
            time.sleep(CONFIG["request_sleep"])
            if details is None:
                continue

            uniprot = details["uniprot"]
            if uniprot in seen_uniprot:
                continue

            length = details["length"]
            if length is None or length < CONFIG["min_length"] or length > CONFIG["max_length"]:
                continue
            if not details["sequence"]:
                continue

            # Drop polyprotein-derived entries before they enter the dataset.
            if is_fragment_risk(uniprot):
                fragments_skipped += 1
                time.sleep(CONFIG["request_sleep"])
                continue
            time.sleep(CONFIG["request_sleep"])

            version = af_model_exists(uniprot)
            if version is None:
                continue

            resolution = fetch_resolution(details["pdb_id"])
            time.sleep(CONFIG["request_sleep"])

            seen_uniprot.add(uniprot)
            candidates.append({
                "name": details["name"],
                "type": type_label,
                "pdb_id": details["pdb_id"],
                "pdb_chain": details["pdb_chain"],
                "uniprot": uniprot,
                "length": length,
                "sequence": details["sequence"],
                "af_version": version,
                "resolution": resolution,
            })
            # Persist after every acceptance: this build is long and is
            # routinely interrupted, and losing it means starting over.
            save_checkpoint(
                [c for c in (prior or []) if c["type"] != type_label] + candidates,
                seen_examined)
            print("    [{0:>3}] {1} {2} ({3})".format(
                len(candidates), details["pdb_id"], uniprot, details["name"][:44]))

        print("  examined {0} new entities, {1} viable {2} candidates"
              " ({3} excluded as AlphaFold fragments).".format(
                  examined, len(candidates), type_label, fragments_skipped))

    except Exception as exc:
        print("  SEARCH ERROR for {0}: {1}".format(type_label, exc))
        print("  Returning {0} partial candidate(s) gathered before the error.".format(
            len(candidates)))

    return candidates


# ---------------------------------------------------------------------------
# Disorder estimation and binning.
# ---------------------------------------------------------------------------
def disorder_bin(disorder_pct):
    """Bucket a disorder percentage into 0-25 / 25-50 / 50-75 / 75-100."""
    if disorder_pct is None:
        return None
    if disorder_pct < 25:
        return "0-25"
    if disorder_pct < 50:
        return "25-50"
    if disorder_pct < 75:
        return "50-75"
    return "75-100"


def _fraction_disordered(scores, threshold):
    """Fraction of per-residue scores at or above threshold (pure Python)."""
    values = list(scores)
    if not values:
        return None
    hits = sum(1 for x in values if float(x) >= threshold)
    return hits / len(values)


def predict_disorder_fractions(sequences):
    """Return a disorder fraction (0 to 1) per sequence, or None where it fails.

    Uses metapredict's batch predictor for speed, falling back to per-sequence
    prediction if the batch call is unavailable.
    """
    threshold = CONFIG["disorder_threshold"]
    fractions = [None] * len(sequences)
    try:
        batch = _metapredict.predict_disorder_batch(sequences)
        for idx, item in enumerate(batch):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                scores = item[1]
            else:
                scores = item
            fractions[idx] = _fraction_disordered(scores, threshold)
        return fractions
    except Exception as exc:
        print("  batch disorder prediction failed ({0}); using per-sequence.".format(exc))

    for idx, seq in enumerate(sequences):
        try:
            scores = _metapredict.predict_disorder(seq)
            if hasattr(scores, "disorder"):
                scores = scores.disorder
            fractions[idx] = _fraction_disordered(scores, threshold)
        except Exception:
            fractions[idx] = None
    return fractions


def assign_disorder(candidates):
    """Attach disorder_frac and bin to each candidate. Return True if computed."""
    if not METAPREDICT_OK or not candidates:
        for c in candidates:
            c["disorder_frac"] = None
            c["bin"] = None
        return False

    print("\nEstimating disorder for {0} candidates with metapredict.".format(
        len(candidates)))
    fractions = predict_disorder_fractions([c["sequence"] for c in candidates])
    for c, frac in zip(candidates, fractions):
        c["disorder_frac"] = frac
        c["bin"] = disorder_bin(frac * 100.0) if frac is not None else None
    scored = sum(1 for c in candidates if c["bin"] is not None)
    print("  disorder scored for {0} of {1} candidates.".format(scored, len(candidates)))
    return scored > 0


# ---------------------------------------------------------------------------
# Balanced selection across disorder bins and types.
# ---------------------------------------------------------------------------
def balanced_select(candidates, target_total):
    """Select proteins to fill the disorder bins as evenly as data allows.

    Each (bin, type) cell is capped at target_total / (bins * types), taken
    symmetrically for viral and cellular so the target is balanced within every
    bin; where the data is short of the cap, the cell simply holds fewer.
    """
    per_cell = max(1, target_total // (len(DISORDER_BINS) * len(GROUP_TYPES)))
    buckets = {(b, t): [] for b in DISORDER_BINS for t in GROUP_TYPES}
    for c in candidates:
        if c.get("bin") in DISORDER_BINS:
            buckets[(c["bin"], c["type"])].append(c)

    selected = []
    for b in DISORDER_BINS:
        for t in GROUP_TYPES:
            selected.extend(buckets[(b, t)][:per_cell])
    return selected, per_cell


def fallback_select(candidates, target_total):
    """Select the first target_total/2 candidates per type (no disorder data)."""
    per_group = max(1, target_total // len(GROUP_TYPES))
    selected = []
    for t in GROUP_TYPES:
        group = [c for c in candidates if c["type"] == t]
        selected.extend(group[:per_group])
    return selected


def print_balance_table(candidates, selected, per_cell):
    """Print selected vs available counts per disorder bin per type."""
    available = {(b, t): 0 for b in DISORDER_BINS for t in GROUP_TYPES}
    chosen = {(b, t): 0 for b in DISORDER_BINS for t in GROUP_TYPES}
    for c in candidates:
        if c.get("bin") in DISORDER_BINS:
            available[(c["bin"], c["type"])] += 1
    for c in selected:
        chosen[(c["bin"], c["type"])] += 1

    print("\nFinal counts per disorder bin per type "
          "(selected / available, cap {0} per cell):".format(per_cell))
    print("  {0:10} {1:>18} {2:>18} {3:>8}".format(
        "bin", "viral", "cellular", "total"))
    for b in DISORDER_BINS:
        v, c = chosen[(b, "viral")], chosen[(b, "cellular")]
        print("  {0:10} {1:>18} {2:>18} {3:>8}".format(
            b,
            "{0} / {1}".format(v, available[(b, "viral")]),
            "{0} / {1}".format(c, available[(b, "cellular")]),
            v + c))
    tv = sum(chosen[(b, "viral")] for b in DISORDER_BINS)
    tc = sum(chosen[(b, "cellular")] for b in DISORDER_BINS)
    print("  {0:10} {1:>18} {2:>18} {3:>8}".format("TOTAL", tv, tc, tv + tc))


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    print("Building protein registry from RCSB.")
    print("Target total {0}, length {1}-{2} aa, resolution <= {3} A.".format(
        CONFIG["target_total"], CONFIG["min_length"],
        CONFIG["max_length"], CONFIG["max_resolution"]))

    print("Excluding entries whose UniProt sequence exceeds {0} aa, which"
          " AlphaFold serves in fragments.".format(CONFIG["af_max_single_fragment"]))

    prior, examined_ids = load_checkpoint()
    if prior:
        print("Resuming from checkpoint: {0} candidate(s), {1} entities already"
              " examined.".format(len(prior), len(examined_ids)))

    candidates = list(prior)
    for type_label, taxonomy_id in CONFIG["taxonomy"].items():
        gathered = gather_candidates(type_label, taxonomy_id, prior=candidates,
                                     examined_ids=examined_ids)
        # gather_candidates returns this group's full set (prior plus new).
        candidates = [c for c in candidates if c["type"] != type_label] + gathered
        save_checkpoint(candidates, examined_ids)

    print("\nCandidate pool: {0} total ({1}).".format(
        len(candidates),
        ", ".join("{0}={1}".format(t, sum(1 for c in candidates if c["type"] == t))
                  for t in GROUP_TYPES)))

    disorder_ok = assign_disorder(candidates)
    if disorder_ok:
        selected, per_cell = balanced_select(candidates, CONFIG["target_total"])
        print_balance_table(candidates, selected, per_cell)
    else:
        print("\nSelecting without disorder balancing (metapredict unavailable).")
        selected = fallback_select(candidates, CONFIG["target_total"])

    out = CONFIG["output_csv"]
    fieldnames = ["name", "type", "pdb_id", "pdb_chain", "uniprot", "resolution"]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected:
            row = {k: record.get(k) for k in fieldnames}
            res = row.get("resolution")
            row["resolution"] = "" if res is None else "{0:.2f}".format(res)
            writer.writerow(row)

    counts = {}
    for record in selected:
        counts[record["type"]] = counts.get(record["type"], 0) + 1
    print("\nWrote {0} proteins to {1}: {2}".format(
        len(selected), out,
        ", ".join("{0}={1}".format(k, v) for k, v in counts.items()) or "none"))


if __name__ == "__main__":
    main()

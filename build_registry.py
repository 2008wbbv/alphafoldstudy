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
     details through the REST data API.
  3. Keep one representative per UniProt accession, inside a length window.
  4. Confirm an AlphaFold model exists before accepting the protein.
  5. Stop at the per-group cap.

All tunables live in the CONFIG dict at the top of the file.

Run with:  python build_registry.py
"""

import csv
import importlib
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Dependency bootstrap: install on first run, fall back to system override.
# ---------------------------------------------------------------------------
def ensure_deps():
    """Make sure third-party imports are available, installing if needed.

    Mapping is import-name -> pip-name. A plain "pip install" is tried first;
    if that is rejected (for example on an externally managed interpreter) we
    retry with --break-system-packages.
    """
    required = {
        "requests": "requests",
        "rcsbapi": "rcsb-api",
    }
    missing = {}
    for import_name, pip_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing[import_name] = pip_name

    if not missing:
        return

    pip_names = sorted(set(missing.values()))
    print("Installing missing dependencies: " + ", ".join(pip_names))
    base = [sys.executable, "-m", "pip", "install", "--quiet"]
    try:
        subprocess.check_call(base + pip_names)
    except subprocess.CalledProcessError:
        print("Standard install failed, retrying with --break-system-packages")
        subprocess.check_call(base + ["--break-system-packages"] + pip_names)
    importlib.invalidate_caches()


ensure_deps()

import requests  # noqa: E402
from rcsbapi.search import AttributeQuery  # noqa: E402


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

    # How many distinct proteins to keep per group, and how many raw search
    # hits to examine before giving up on reaching that cap. pool_size bounds
    # the number of REST round-trips so a run finishes in reasonable time.
    # Note: distinct viral proteins are sparser near the front of the search
    # order (a few heavily-deposited proteins dominate and collapse under the
    # per-accession dedup), so filling the viral cap can need a larger pool than
    # the cellular cap. Raise pool_size if the viral group comes up short.
    "per_group_cap": 30,
    "pool_size": 600,

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
    "af_url": "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_{version}.pdb",
    "af_api_url": "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}",

    # Politeness and robustness.
    "request_sleep": 0.1,
    "request_timeout": 30,

    "output_csv": "proteins.csv",
}


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

    Returns a dict with keys: pdb_id, pdb_chain, uniprot, name, length.
    Returns None when the entity has no UniProt mapping or the lookup fails.
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

    return {
        "pdb_id": pdb.upper(),
        "pdb_chain": auth_asym_ids[0],
        "uniprot": uniprot_ids[0],
        "name": poly_entity.get("pdbx_description") or "Unknown",
        "length": entity_poly.get("rcsb_sample_sequence_length"),
    }


# ---------------------------------------------------------------------------
# AlphaFold model existence check.
# ---------------------------------------------------------------------------
def af_model_exists(uniprot):
    """Return a tag for the AlphaFold model source if one exists, else None.

    Tries the classic UniProt-accession file URL first (the URL named in the
    study brief), then the AlphaFold API as a fall-back.
    """
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

    if CONFIG["use_af_api_fallback"]:
        url = CONFIG["af_api_url"].format(uniprot=uniprot)
        try:
            resp = requests.get(url, timeout=CONFIG["request_timeout"])
            if resp.status_code == 200 and resp.json():
                return "api"
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Select one group.
# ---------------------------------------------------------------------------
def select_group(type_label, taxonomy_id):
    """Select up to per_group_cap proteins for a single group.

    Always returns whatever was gathered, even if the search raises partway
    through, so a drifted attribute name produces partial results rather than
    a silent failure.
    """
    print("\n=== Selecting {0} proteins (taxonomy {1}) ===".format(
        type_label, taxonomy_id))
    selected = []
    seen_uniprot = set()

    try:
        query = build_query(taxonomy_id)
        results = query(return_type="polymer_entity")

        examined = 0
        for entity_id in results:
            if len(selected) >= CONFIG["per_group_cap"]:
                break
            if examined >= CONFIG["pool_size"]:
                break
            examined += 1

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

            version = af_model_exists(uniprot)
            if version is None:
                continue

            seen_uniprot.add(uniprot)
            record = {
                "name": details["name"],
                "type": type_label,
                "pdb_id": details["pdb_id"],
                "pdb_chain": details["pdb_chain"],
                "uniprot": uniprot,
            }
            selected.append(record)
            print("  [{0:>8}] {1}_{2} {3} ({4} aa) AF:{5}  {6}".format(
                type_label, record["pdb_id"], record["pdb_chain"],
                uniprot, length, version, record["name"]))

    except Exception as exc:
        print("  SEARCH ERROR for {0}: {1}".format(type_label, exc))
        print("  Returning {0} partial result(s) gathered before the error.".format(
            len(selected)))

    print("  -> {0} {1} proteins selected.".format(len(selected), type_label))
    return selected


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    print("Building protein registry from RCSB.")
    print("Caps: {0} per group, length {1}-{2} aa, resolution <= {3} A.".format(
        CONFIG["per_group_cap"], CONFIG["min_length"],
        CONFIG["max_length"], CONFIG["max_resolution"]))

    all_records = []
    for type_label, taxonomy_id in CONFIG["taxonomy"].items():
        all_records.extend(select_group(type_label, taxonomy_id))

    out = CONFIG["output_csv"]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["name", "type", "pdb_id", "pdb_chain", "uniprot"])
        writer.writeheader()
        for record in all_records:
            writer.writerow(record)

    counts = {}
    for record in all_records:
        counts[record["type"]] = counts.get(record["type"], 0) + 1
    print("\nWrote {0} proteins to {1}: {2}".format(
        len(all_records), out,
        ", ".join("{0}={1}".format(k, v) for k, v in counts.items()) or "none"))


if __name__ == "__main__":
    main()

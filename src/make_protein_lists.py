#!/usr/bin/env python3
"""make_protein_lists.py

Generate ready-to-use protein registries so a fresh clone can run immediately.

This does the same job as build_registry.py but is built for scale. Instead of
one REST round trip per candidate, it fetches entity details from the RCSB
GraphQL endpoint in batches of 150, looks up UniProt lengths in batches of 100,
and checks AlphaFold model existence in parallel. Building a thousand-protein
registry drops from hours to a couple of minutes.

Written registries (committed to the repo so nobody has to rebuild them):

    proteins_250.csv     250 proteins, balanced viral and cellular
    proteins_1000.csv    1000 proteins, balanced viral and cellular

Selection rules, matching the study design:

  - protein entities solved at 3.0 A resolution or better
  - viral (taxonomy 10239) and human (taxonomy 9606)
  - 50 to 600 residues in the crystallised entity
  - one representative per UniProt accession
  - an AlphaFold model must exist
  - the parent UniProt sequence must not exceed the AlphaFold single-model
    limit, which excludes the polyprotein fragments that would otherwise
    produce artificially low scores

Run with:  python src/make_protein_lists.py
           python src/make_protein_lists.py --sizes 250 1000 --pool 12000
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from rcsbapi.search import AttributeQuery

CONFIG = {
    "taxonomy": {"viral": "10239", "cellular": "9606"},
    "polymer_type": "Protein",
    "max_resolution": 3.0,
    "min_length": 50,
    "max_length": 600,
    # Longer parent sequences are served by AlphaFold in fragments, so the
    # downloaded model may not contain the crystallised region at all.
    "af_max_single_fragment": 2700,

    "graphql_url": "https://data.rcsb.org/graphql",
    "uniprot_search_url": "https://rest.uniprot.org/uniprotkb/search",
    "af_api_url": "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}",

    "entity_batch": 150,     # entity ids per GraphQL request
    "uniprot_batch": 100,    # accessions per UniProt request
    "af_workers": 16,        # parallel AlphaFold existence checks
    "timeout": 90,
}

ENTITY_FIELDS = """
  rcsb_id
  rcsb_polymer_entity_container_identifiers { uniprot_ids auth_asym_ids }
  entity_poly { rcsb_sample_sequence_length }
  rcsb_polymer_entity { pdbx_description }
  entry { rcsb_entry_info { resolution_combined } }
"""

FIELDNAMES = ["name", "type", "pdb_id", "pdb_chain", "uniprot", "resolution"]


def search_entity_ids(taxonomy_id, limit):
    """Return up to `limit` polymer-entity IDs for one taxonomic group."""
    q_tax = AttributeQuery(
        "rcsb_entity_source_organism.taxonomy_lineage.id", "exact_match", taxonomy_id)
    q_type = AttributeQuery(
        "entity_poly.rcsb_entity_polymer_type", "exact_match", CONFIG["polymer_type"])
    q_res = AttributeQuery(
        "rcsb_entry_info.resolution_combined", "less_or_equal", CONFIG["max_resolution"])

    ids = []
    for entity_id in (q_tax & q_type & q_res)(return_type="polymer_entity"):
        ids.append(entity_id)
        if len(ids) >= limit:
            break
    return ids


def fetch_entities(entity_ids):
    """Fetch entity details from GraphQL in batches. Returns a list of dicts."""
    out = []
    size = CONFIG["entity_batch"]
    for start in range(0, len(entity_ids), size):
        chunk = entity_ids[start:start + size]
        id_list = ",".join('"{0}"'.format(i) for i in chunk)
        query = "{ polymer_entities(entity_ids: [%s]) { %s } }" % (id_list, ENTITY_FIELDS)
        try:
            resp = requests.post(CONFIG["graphql_url"], json={"query": query},
                                 timeout=CONFIG["timeout"])
            entities = (resp.json().get("data") or {}).get("polymer_entities") or []
        except Exception as exc:
            print("  batch at {0} failed: {1}".format(start, exc))
            continue
        out.extend(e for e in entities if e)
        sys.stdout.write("\r  fetched {0}/{1} entities".format(len(out), len(entity_ids)))
        sys.stdout.flush()
    print()
    return out


def parse_entity(entity, type_label):
    """Turn one GraphQL entity into a candidate dict, or None if unusable."""
    ident = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    uniprot_ids = ident.get("uniprot_ids") or []
    chains = ident.get("auth_asym_ids") or []
    if not uniprot_ids or not chains:
        return None

    poly = entity.get("entity_poly") or {}
    length = poly.get("rcsb_sample_sequence_length")
    if length is None or length < CONFIG["min_length"] or length > CONFIG["max_length"]:
        return None

    resolutions = (((entity.get("entry") or {}).get("rcsb_entry_info") or {})
                   .get("resolution_combined") or [])
    if not resolutions:
        return None

    rcsb_id = entity.get("rcsb_id") or ""
    pdb_id = rcsb_id.split("_")[0].upper()
    if not pdb_id:
        return None

    description = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description")
    return {
        "name": (description or "Unknown").replace("\n", " ").strip(),
        "type": type_label,
        "pdb_id": pdb_id,
        "pdb_chain": chains[0],
        "uniprot": uniprot_ids[0],
        "resolution": "{0:.2f}".format(float(resolutions[0])),
    }


def uniprot_lengths(accessions):
    """Map accession -> sequence length, fetched in batches."""
    lengths = {}
    accessions = list(accessions)
    size = CONFIG["uniprot_batch"]
    for start in range(0, len(accessions), size):
        chunk = accessions[start:start + size]
        query = " OR ".join("accession:{0}".format(a) for a in chunk)
        try:
            resp = requests.get(CONFIG["uniprot_search_url"],
                                params={"query": query, "fields": "accession,length",
                                        "format": "tsv", "size": 500},
                                timeout=CONFIG["timeout"])
            for line in resp.text.strip().split("\n")[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        lengths[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
        except Exception as exc:
            print("  uniprot batch at {0} failed: {1}".format(start, exc))
        sys.stdout.write("\r  resolved {0}/{1} UniProt lengths".format(
            len(lengths), len(accessions)))
        sys.stdout.flush()
    print()
    return lengths


def af_model_exists(uniprot):
    """True when the AlphaFold database serves a model for this accession."""
    try:
        resp = requests.get(CONFIG["af_api_url"].format(uniprot=uniprot), timeout=30)
        return resp.status_code == 200 and bool(resp.json())
    except Exception:
        return False


def filter_alphafold(candidates):
    """Keep only candidates that have an AlphaFold model, checked in parallel."""
    accessions = [c["uniprot"] for c in candidates]
    with ThreadPoolExecutor(max_workers=CONFIG["af_workers"]) as pool:
        found = list(pool.map(af_model_exists, accessions))
    kept = [c for c, ok in zip(candidates, found) if ok]
    print("  {0} of {1} have AlphaFold models".format(len(kept), len(candidates)))
    return kept


def gather(type_label, taxonomy_id, pool_size):
    """Build the full candidate list for one group."""
    print("\n=== {0} (taxonomy {1}) ===".format(type_label, taxonomy_id))

    print("  searching RCSB for up to {0} entities".format(pool_size))
    entity_ids = search_entity_ids(taxonomy_id, pool_size)
    print("  search returned {0}".format(len(entity_ids)))

    entities = fetch_entities(entity_ids)

    # Parse, then keep one representative per accession so that heavily
    # deposited proteins cannot dominate the sample.
    seen = set()
    candidates = []
    for entity in entities:
        parsed = parse_entity(entity, type_label)
        if parsed is None or parsed["uniprot"] in seen:
            continue
        seen.add(parsed["uniprot"])
        candidates.append(parsed)
    print("  {0} distinct accessions inside the length window".format(len(candidates)))

    lengths = uniprot_lengths(c["uniprot"] for c in candidates)
    limit = CONFIG["af_max_single_fragment"]
    before = len(candidates)
    # An accession whose length is unknown is kept; the coverage check in
    # af_study.py is the backstop for anything that slips through.
    candidates = [c for c in candidates if lengths.get(c["uniprot"], 0) <= limit]
    print("  {0} excluded as AlphaFold fragments (parent over {1} aa)".format(
        before - len(candidates), limit))

    return filter_alphafold(candidates)


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDNAMES})
    counts = {}
    for row in rows:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    print("  {0}: {1} proteins ({2})".format(
        path, len(rows),
        ", ".join("{0}={1}".format(k, v) for k, v in sorted(counts.items()))))


def main():
    parser = argparse.ArgumentParser(description="Generate ready-made protein registries.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000],
                        help="registry sizes to write (default: 250 1000)")
    parser.add_argument("--pool", type=int, default=12000,
                        help="max search hits to examine per group (default: 12000)")
    args = parser.parse_args()

    pools = {}
    for type_label, taxonomy_id in CONFIG["taxonomy"].items():
        pools[type_label] = gather(type_label, taxonomy_id, args.pool)

    print("\n=== available ===")
    for type_label, rows in pools.items():
        print("  {0}: {1}".format(type_label, len(rows)))

    # An accession can legitimately turn up in both searches, for example a
    # human protein solved as part of a viral complex, since the taxonomy
    # filter matches any source organism on the entry. Deduplicating only
    # within each group would then let the same protein appear twice in the
    # combined file, which breaks the one-representative-per-accession rule the
    # whole design rests on. Resolve the overlap once, in favour of the group
    # that has fewer candidates to spare.
    viral_accessions = {c["uniprot"] for c in pools["viral"]}
    overlap = [c for c in pools["cellular"] if c["uniprot"] in viral_accessions]
    if overlap:
        pools["cellular"] = [c for c in pools["cellular"]
                             if c["uniprot"] not in viral_accessions]
        print("\n  {0} accession(s) present in both groups, kept as viral".format(
            len(overlap)))

    print("\n=== writing registries ===")
    for size in sorted(args.sizes):
        half = size // 2
        # Take an even split where possible. If one group is short, top up from
        # the other so the file still holds the requested number of proteins,
        # and say so, since the balance matters for the comparison.
        viral = pools["viral"][:half]
        cellular = pools["cellular"][:half]
        shortfall = size - len(viral) - len(cellular)
        if shortfall > 0:
            extra_cell = pools["cellular"][len(cellular):len(cellular) + shortfall]
            cellular = cellular + extra_cell
            shortfall -= len(extra_cell)
        if shortfall > 0:
            extra_viral = pools["viral"][len(viral):len(viral) + shortfall]
            viral = viral + extra_viral

        rows = viral + cellular
        accessions = {r["uniprot"] for r in rows}
        if len(accessions) != len(rows):
            print("  WARNING: {0} duplicate accession(s) in {1}".format(
                len(rows) - len(accessions), size))
        if len(rows) < size:
            print("  note: only {0} of {1} available".format(len(rows), size))
        if len(viral) != len(cellular):
            print("  note: groups are uneven ({0} viral, {1} cellular);"
                  " the limiting group is capped by what the PDB contains".format(
                      len(viral), len(cellular)))
        write_csv("proteins_{0}.csv".format(size), rows)

    print("\nDone. Point the study at one of them:")
    print("  python src/af_study.py --registry proteins_250.csv")


if __name__ == "__main__":
    main()

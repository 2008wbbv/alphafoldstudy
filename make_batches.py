#!/usr/bin/env python3
"""make_batches.py

Split a protein registry (proteins.csv by default) into fixed-size batches for
distribution across an HPC cluster, and emit a matching SLURM job script for
each batch. This only generates files; it never submits or runs anything.

Before batching, rows are pre-filtered to structures with a usable resolution:
a row is dropped if no resolution can be found for it, or if the resolution is
worse (numerically greater) than max_resolution. Resolution is taken from a
column in the registry CSV if one is present, otherwise it is read from the
already-downloaded structure file in data/ (so this works offline, with no
re-download). If neither source has it, the row is treated as missing.

Run with:  python make_batches.py [--input proteins.csv] [--batch-size 250]
"""

import argparse
import csv
import math
import os
import re

CONFIG = {
    "input_csv": "proteins.csv",
    "out_dir": "batches",
    "log_dir": "logs",
    "data_dir": "data",
    "batch_size": 250,
    "max_resolution": 3.0,
    # Column names that may carry a resolution value in the registry CSV.
    "resolution_columns": ["resolution", "resolution_combined", "resolution_A"],
    # Default SLURM header values. These are written into every job script and
    # are meant to be edited per cluster.
    "slurm": {
        "time": "12:00:00",
        "mem": "16G",
        "cpus": "4",
        "job_prefix": "afstudy",
    },
}

REGISTRY_FIELDS = ["name", "type", "pdb_id", "pdb_chain", "uniprot"]


def parse_resolution_from_file(path):
    """Read a resolution (Angstrom) from a local PDB or mmCIF file, or None."""
    try:
        with open(path, errors="ignore") as handle:
            text = handle.read()
    except Exception:
        return None
    if path.endswith(".pdb"):
        match = re.search(r"REMARK\s+2 RESOLUTION\.\s+([\d.]+)\s+ANGSTROM", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    else:
        # X-ray keys first, then the cryo-EM reconstruction resolution so EM
        # structures (which carry no diffraction resolution) are not lost.
        for key in ("_refine.ls_d_res_high", "_reflns.d_resolution_high",
                    "_em_3d_reconstruction.resolution"):
            match = re.search(re.escape(key) + r"\s+([\d.]+)", text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    return None


def get_resolution(row, data_dir, res_columns):
    """Resolution for a registry row: CSV column first, then local structure file."""
    for col in res_columns:
        value = row.get(col)
        if value not in (None, "", "NA"):
            try:
                return float(value)
            except ValueError:
                pass
    pdb = (row.get("pdb_id") or "").strip()
    if pdb:
        for ext in (".pdb", ".cif"):
            path = os.path.join(data_dir, pdb + ext)
            if os.path.exists(path):
                resolution = parse_resolution_from_file(path)
                if resolution is not None:
                    return resolution
    return None


def slurm_script(job_name, batch_csv, results_dir, log_dir, cfg):
    """Return the text of a SLURM job script for one batch."""
    return """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_dir}/{job}_%j.log

# ==========================================================================
# EDIT THESE CLUSTER-SPECIFIC SETTINGS BEFORE SUBMITTING:
#   1. Account and partition (uncomment and fill in):
#        #SBATCH --account=YOUR_ACCOUNT
#        #SBATCH --partition=YOUR_PARTITION
#   2. Module / environment setup (uncomment and edit for your cluster):
#        module load python/3.11
#        source /path/to/your/venv/bin/activate
#   3. Confirm the working directory and paths below are correct.
# The #SBATCH headers above (job name, time, memory, cpus, output log) are
# safe defaults; tune them to your job and cluster limits.
# ==========================================================================

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-.}}"
mkdir -p {results_dir} {log_dir}

# module load python/3.11                 # <- edit for your cluster
# source /path/to/your/venv/bin/activate  # <- edit for your cluster

python af_study.py --registry {batch_csv} --results-dir {results_dir}
""".format(
        job=job_name,
        time=cfg["time"],
        mem=cfg["mem"],
        cpus=cfg["cpus"],
        log_dir=log_dir,
        results_dir=results_dir,
        batch_csv=batch_csv,
    )


def main():
    parser = argparse.ArgumentParser(description="Split a registry into HPC batches.")
    parser.add_argument("--input", default=CONFIG["input_csv"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--max-resolution", type=float, default=CONFIG["max_resolution"])
    parser.add_argument("--out-dir", default=CONFIG["out_dir"])
    args = parser.parse_args()

    with open(args.input, newline="") as handle:
        rows = list(csv.DictReader(handle))
    print("Read {0} rows from {1}.".format(len(rows), args.input))

    kept, dropped_missing, dropped_res = [], 0, 0
    for row in rows:
        resolution = get_resolution(row, CONFIG["data_dir"], CONFIG["resolution_columns"])
        if resolution is None:
            dropped_missing += 1
            continue
        if resolution > args.max_resolution:
            dropped_res += 1
            continue
        row["resolution"] = "{0:.2f}".format(resolution)
        kept.append(row)
    print("Kept {0}; dropped {1} (missing resolution), {2} (worse than {3} A).".format(
        len(kept), dropped_missing, dropped_res, args.max_resolution))

    if not kept:
        print("No rows survived the resolution filter; nothing to batch.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    n_batches = math.ceil(len(kept) / args.batch_size)
    pad = max(2, len(str(n_batches)))
    fields = REGISTRY_FIELDS + ["resolution"]

    for index in range(n_batches):
        chunk = kept[index * args.batch_size:(index + 1) * args.batch_size]
        tag = "batch_{0}".format(str(index + 1).zfill(pad))
        csv_path = os.path.join(args.out_dir, tag + ".csv")
        sh_path = os.path.join(args.out_dir, tag + ".sh")

        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in chunk:
                writer.writerow({k: row.get(k, "") for k in fields})

        job_name = "{0}_{1}".format(CONFIG["slurm"]["job_prefix"], tag)
        results_dir = os.path.join("results", tag)
        with open(sh_path, "w") as handle:
            handle.write(slurm_script(
                job_name, csv_path, results_dir, CONFIG["log_dir"], CONFIG["slurm"]))

        print("  wrote {0} ({1} proteins) and {2}".format(csv_path, len(chunk), sh_path))

    print("Generated {0} batch(es) of up to {1} proteins in {2}/.".format(
        n_batches, args.batch_size, args.out_dir))
    print("Nothing was submitted or executed. Edit the .sh headers, then sbatch them.")


if __name__ == "__main__":
    main()

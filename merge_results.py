#!/usr/bin/env python3
"""merge_results.py

Concatenate the per-batch outputs produced by HPC runs of af_study.py back into
single combined tables. Each batch writes results/batch_XX/metrics.csv and
results/batch_XX/per_residue.csv (see make_batches.py); this script stitches
them into results/combined_metrics.csv and results/combined_per_residue.csv.

It only reads and concatenates existing files; it runs no analysis.

Run with:  python merge_results.py
"""

import csv
import glob
import os

RESULTS = "results"


def merge(filename, out_name):
    """Concatenate results/batch_*/<filename> into results/<out_name>."""
    paths = sorted(glob.glob(os.path.join(RESULTS, "batch_*", filename)))
    if not paths:
        print("No {0} files found under {1}/batch_*/; skipping.".format(filename, RESULTS))
        return

    out_path = os.path.join(RESULTS, out_name)
    header = None
    total = 0
    with open(out_path, "w", newline="") as out_handle:
        writer = None
        for path in paths:
            with open(path, newline="") as in_handle:
                reader = csv.reader(in_handle)
                rows = list(reader)
            if not rows:
                print("  {0}: empty, skipped.".format(path))
                continue
            file_header, data = rows[0], rows[1:]
            if header is None:
                header = file_header
                writer = csv.writer(out_handle)
                writer.writerow(header)
            elif file_header != header:
                print("  WARNING: header mismatch in {0}; skipping this file.".format(path))
                continue
            writer.writerows(data)
            total += len(data)
            print("  {0}: {1} rows".format(path, len(data)))

    print("Wrote {0} ({1} rows from {2} file(s)).".format(out_path, total, len(paths)))


def main():
    print("Merging per-batch results under {0}/ ...".format(RESULTS))
    merge("metrics.csv", "combined_metrics.csv")
    merge("per_residue.csv", "combined_per_residue.csv")


if __name__ == "__main__":
    main()

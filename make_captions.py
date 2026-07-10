#!/usr/bin/env python3
"""make_captions.py

Write a short caption for every figure already in results/, pulling the numbers
straight from the existing stat files (results/stats.txt,
results/per_residue_stats.txt). Nothing is recomputed and nothing is invented:
if a required number cannot be found in the files, the caption prints
"STAT MISSING" in its place.

Reads only from disk. Run with:  python make_captions.py
"""

import os
import re

RESULTS = "results"
MISSING = "STAT MISSING"


def read(path):
    try:
        with open(path) as handle:
            return handle.read()
    except Exception:
        return ""


def search(pattern, text, groups):
    """Return the requested capture groups, or MISSING placeholders."""
    if text:
        match = re.search(pattern, text)
        if match:
            return match.groups()
    return tuple(MISSING for _ in range(groups))


def main():
    stats = read(os.path.join(RESULTS, "stats.txt"))
    per_residue = read(os.path.join(RESULTS, "per_residue_stats.txt"))

    # The figures are built from the coverage-filtered set, so read that block.
    cov = ""
    if "COVERAGE-FILTERED" in stats:
        cov = stats[stats.index("COVERAGE-FILTERED"):]

    med_v, n_v, med_c, n_c, mw_p = search(
        r"Mann-Whitney U on TM-score \(viral vs cellular\)\s*\n"
        r"\s*median viral\s*=\s*([-\d.]+) \(n=(\d+)\)\s*\n"
        r"\s*median cellular\s*=\s*([-\d.]+) \(n=(\d+)\)\s*\n"
        r"\s*U = [-\d.]+, p = ([\d.eE+-]+)", cov, 5)

    plddt_rho, plddt_p, plddt_n = search(
        r"Spearman: pLDDT vs TM-score \(overall\)\s*\n"
        r"\s*rho = ([+\-\d.]+), p = ([\d.eE+-]+) \(n=(\d+)\)", cov, 3)

    dis_rho, dis_p, dis_n = search(
        r"Spearman: disorder% vs TM-score \(overall\)\s*\n"
        r"\s*rho = ([+\-\d.]+), p = ([\d.eE+-]+) \(n=(\d+)\)", cov, 3)

    pr_rho, pr_p, pr_n = search(
        r"Spearman: disorder score vs CA distance \(all residues\)\s*\n"
        r"\s*rho = ([+\-\d.]+), p = ([\d.eE+-]+) \(n=(\d+) residues\)", per_residue, 3)

    dmed, dn, omed, on_, prmw_p = search(
        r"median disordered = ([\d.]+) A \(n=(\d+)\)\s*\n"
        r"\s*median ordered\s*=\s*([\d.]+) A \(n=(\d+)\)\s*\n"
        r"\s*Mann-Whitney U = [\d.]+, p = ([\d.eE+-]+)", per_residue, 5)

    kw_bins, kw_h, kw_p, kw_nbins = search(
        r"Kruskal-Wallis: TM-score across disorder bins\s*\n"
        r"((?:\s*bin .*\n)+?)\s*H = ([\d.]+), p = ([\d.eE+-]+) across (\d+) bins", cov, 4)
    if kw_bins != MISSING:
        kw_bins = " ".join(part.strip() for part in kw_bins.strip().splitlines())

    captions = []
    captions.append((
        "tm_boxplot.png",
        "TM-score (AlphaFold vs experimental) by protein type, coverage-filtered "
        "set. Median TM-score is {0} for viral (n={1}) and {2} for cellular "
        "(n={3}); the viral vs cellular difference is not significant "
        "(Mann-Whitney p = {4}).".format(med_v, n_v, med_c, n_c, mw_p)))

    captions.append((
        "plddt_vs_tm.png",
        "Per-protein mean pLDDT vs TM-score, colored by type with fragment-flagged "
        "proteins drawn as x markers (coverage-filtered set; the correlation is "
        "over the kept proteins). Higher AlphaFold confidence tracks higher "
        "accuracy: Spearman rho = {0}, p = {1} (n={2}).".format(
            plddt_rho, plddt_p, plddt_n)))

    captions.append((
        "disorder_vs_tm.png",
        "Predicted disorder (percent) vs TM-score, colored by type with "
        "fragment-flagged proteins drawn as x markers (coverage-filtered set). "
        "More disorder goes with lower accuracy: Spearman rho = {0}, p = {1} "
        "(n={2}).".format(dis_rho, dis_p, dis_n)))

    captions.append((
        "per_residue_disorder_hexbin.png",
        "Per-residue local error (CA-CA distance after TM-align superposition) vs "
        "per-residue disorder score, over all status-ok proteins (not "
        "coverage-filtered). Disorder barely relates to local error (Spearman "
        "rho = {0}, p = {1}, n={2}), and disordered residues (median {3} A, "
        "n={4}) do not differ from ordered residues (median {5} A, n={6}; "
        "Mann-Whitney p = {7}).".format(
            pr_rho, pr_p, pr_n, dmed, dn, omed, on_, prmw_p)))

    captions.append((
        "disorder-bin figure",
        "No dedicated disorder-bin figure exists in results/; the disorder-bin "
        "comparison is reported in stats.txt only. For reference "
        "(coverage-filtered): Kruskal-Wallis H = {0}, p = {1} across {2} bins "
        "({3}).".format(kw_h, kw_p, kw_nbins, kw_bins)))

    lines = ["Figure captions (numbers pulled from results/stats.txt and",
             "results/per_residue_stats.txt; STAT MISSING means the value was",
             "not found in those files).", ""]
    for name, text in captions:
        lines.append("[{0}]".format(name))
        lines.append(text)
        lines.append("")

    out = os.path.join(RESULTS, "figure_captions.txt")
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("Wrote {0}.".format(out))


if __name__ == "__main__":
    main()

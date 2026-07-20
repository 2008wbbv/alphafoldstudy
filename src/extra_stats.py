#!/usr/bin/env python3
"""extra_stats.py

Additional statistical tests on the coverage-filtered set already stored in
results/metrics.csv. The new results are appended to results/stats.txt in a
clearly labeled block; the numbers already in that file are never modified.

Reads only from disk (no re-download, no re-run of the pipeline). Appending is
idempotent: if the block is already present the tests are recomputed and
printed but not appended a second time.

Run with:  python extra_stats.py
"""

import csv
import importlib
import importlib.util
import os
import subprocess
import sys


def ensure_deps():
    """Install numpy and scipy if missing, with a system-override fall-back."""
    required = {"numpy": "numpy", "scipy": "scipy"}
    missing = [pip for imp, pip in required.items()
               if importlib.util.find_spec(imp) is None]
    if not missing:
        return
    base = [sys.executable, "-m", "pip", "install", "--quiet"]
    try:
        subprocess.check_call(base + missing)
    except subprocess.CalledProcessError:
        print("Standard install failed, retrying with --break-system-packages")
        subprocess.check_call(base + ["--break-system-packages"] + missing)
    importlib.invalidate_caches()


ensure_deps()

import numpy as np  # noqa: E402
from scipy.stats import (  # noqa: E402
    shapiro, ttest_ind, mannwhitneyu, fisher_exact)

METRICS = os.path.join("results", "metrics.csv")
STATS = os.path.join("results", "stats.txt")
SENTINEL = "EXTRA STATISTICAL TESTS (appended by extra_stats.py)"
FAIL_TM = 0.5
DISORDER_CUT = 0.5
BINS = ["0-25", "25-50", "50-75", "75-100"]


def load_filtered():
    """Load the coverage-filtered, status-ok rows from metrics.csv."""
    rows = []
    with open(METRICS, newline="") as handle:
        for r in csv.DictReader(handle):
            if r["status"] != "ok" or r["fragment_flag"] != "False":
                continue
            if r["tm_score"] == "NA":
                continue
            rows.append(r)
    return rows


def num(value):
    return None if value in ("NA", "") else float(value)


def fmt(value, nd=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if value != value:  # NaN
            return "NA"
        if value in (float("inf"), float("-inf")):
            return "inf"
        return "{0:.{1}g}".format(value, nd)
    return str(value)


def main():
    rows = load_filtered()
    lines = []

    def emit(text=""):
        lines.append(text)

    viral = [r for r in rows if r["type"] == "viral"]
    cellular = [r for r in rows if r["type"] == "cellular"]
    v_tm = [num(r["tm_score"]) for r in viral]
    c_tm = [num(r["tm_score"]) for r in cellular]
    v_rmsd = [num(r["rmsd"]) for r in viral if num(r["rmsd"]) is not None]
    c_rmsd = [num(r["rmsd"]) for r in cellular if num(r["rmsd"]) is not None]

    emit("")
    emit("=" * 60)
    emit(SENTINEL)
    emit("Coverage-filtered set: {0} proteins ({1} viral, {2} cellular).".format(
        len(rows), len(viral), len(cellular)))
    emit("These tests were added after the fact; the blocks above are unchanged.")
    emit("Recomputed from the rounded values in metrics.csv, so the last digits")
    emit("can differ from the full-precision values in the blocks above.")
    emit("=" * 60)
    emit("")

    # 1. Shapiro-Wilk normality on TM-score.
    emit("Shapiro-Wilk normality test on TM-score")
    for label, data in [("viral", v_tm), ("cellular", c_tm)]:
        if len(data) >= 3:
            w_stat, p_value = shapiro(data)
            verdict = ("not normal (reject normality)" if p_value < 0.05
                       else "consistent with normality")
            emit("  {0:9} W = {1}, p = {2} -> {3}".format(
                label, fmt(w_stat), fmt(p_value), verdict))
        else:
            emit("  {0:9} too few points for the test (n={1}).".format(label, len(data)))
    emit("")

    # 2. Welch's t-test next to the existing Mann-Whitney.
    emit("Welch t-test (parametric) next to Mann-Whitney (rank-based)")
    for metric, v_data, c_data in [("TM-score", v_tm, c_tm), ("RMSD", v_rmsd, c_rmsd)]:
        if len(v_data) >= 2 and len(c_data) >= 2:
            t_stat, welch_p = ttest_ind(v_data, c_data, equal_var=False)
            _, mw_p = mannwhitneyu(v_data, c_data, alternative="two-sided")
            agree = (welch_p < 0.05) == (mw_p < 0.05)
            emit("  {0}: Welch t = {1}, Welch p = {2}; Mann-Whitney p = {3}".format(
                metric, fmt(t_stat), fmt(welch_p), fmt(mw_p)))
            emit("    the two tests {0} at alpha 0.05.".format(
                "agree" if agree else "DISAGREE"))
        else:
            emit("  {0}: too few points for the tests.".format(metric))
    emit("")

    # 3. Fisher exact tests with failure defined as TM-score < 0.5.
    emit("Fisher exact test, failure = TM-score < {0}".format(FAIL_TM))

    def fisher_block(title, group_a_label, group_b_label, in_group_a):
        a_fail = sum(1 for r in rows if in_group_a(r) and num(r["tm_score"]) < FAIL_TM)
        a_pass = sum(1 for r in rows if in_group_a(r) and num(r["tm_score"]) >= FAIL_TM)
        b_fail = sum(1 for r in rows if (not in_group_a(r)) and num(r["tm_score"]) < FAIL_TM)
        b_pass = sum(1 for r in rows if (not in_group_a(r)) and num(r["tm_score"]) >= FAIL_TM)
        odds, p_value = fisher_exact([[a_fail, a_pass], [b_fail, b_pass]])
        emit("  {0}".format(title))
        emit("    {0:14} fail={1:<3} pass={2}".format(group_a_label, a_fail, a_pass))
        emit("    {0:14} fail={1:<3} pass={2}".format(group_b_label, b_fail, b_pass))
        emit("    odds ratio = {0}, p = {1}".format(fmt(odds), fmt(p_value)))

    fisher_block("failure vs viral/cellular", "viral", "cellular",
                 lambda r: r["type"] == "viral")
    # Only rows with a disorder value can be classed disordered/ordered.
    dis_rows = [r for r in rows if num(r["disorder_frac"]) is not None]
    if dis_rows:
        def dis_fisher(title):
            a_fail = sum(1 for r in dis_rows if num(r["disorder_frac"]) >= DISORDER_CUT
                         and num(r["tm_score"]) < FAIL_TM)
            a_pass = sum(1 for r in dis_rows if num(r["disorder_frac"]) >= DISORDER_CUT
                         and num(r["tm_score"]) >= FAIL_TM)
            b_fail = sum(1 for r in dis_rows if num(r["disorder_frac"]) < DISORDER_CUT
                         and num(r["tm_score"]) < FAIL_TM)
            b_pass = sum(1 for r in dis_rows if num(r["disorder_frac"]) < DISORDER_CUT
                         and num(r["tm_score"]) >= FAIL_TM)
            odds, p_value = fisher_exact([[a_fail, a_pass], [b_fail, b_pass]])
            emit("  {0}".format(title))
            emit("    {0:14} fail={1:<3} pass={2}".format("disordered", a_fail, a_pass))
            emit("    {0:14} fail={1:<3} pass={2}".format("ordered", b_fail, b_pass))
            emit("    odds ratio = {0}, p = {1}".format(fmt(odds), fmt(p_value)))
        dis_fisher("failure vs disordered/ordered (disorder_pred >= {0})".format(DISORDER_CUT))
    else:
        emit("  disordered/ordered table skipped (no disorder values).")
    emit("")

    # 4. Winsorized re-run (clip TM-score at 5th/95th percentiles).
    emit("Winsorized re-run: TM-score clipped at 5th/95th percentiles")
    all_tm = np.array(v_tm + c_tm, dtype=float)
    lo, hi = np.percentile(all_tm, [5, 95])
    emit("  clip bounds: [{0}, {1}]".format(fmt(float(lo)), fmt(float(hi))))
    v_clip = np.clip(np.array(v_tm, dtype=float), lo, hi)
    c_clip = np.clip(np.array(c_tm, dtype=float), lo, hi)
    if len(v_tm) >= 2 and len(c_tm) >= 2:
        _, mw_raw = mannwhitneyu(v_tm, c_tm, alternative="two-sided")
        _, mw_w = mannwhitneyu(v_clip, c_clip, alternative="two-sided")
        _, t_raw = ttest_ind(v_tm, c_tm, equal_var=False)
        _, t_w = ttest_ind(v_clip, c_clip, equal_var=False)
        emit("  Mann-Whitney p: raw = {0}, winsorized = {1}".format(fmt(mw_raw), fmt(mw_w)))
        emit("  Welch t p:      raw = {0}, winsorized = {1}".format(fmt(t_raw), fmt(t_w)))
        emit("  (rank-based Mann-Whitney barely moves; a large shift in Welch p")
        emit("  would show the low-TM tail is driving the parametric result.)")
    else:
        emit("  too few points for the winsorized tests.")
    emit("")

    # 5. Pairwise Mann-Whitney between adjacent disorder bins, Bonferroni.
    emit("Pairwise Mann-Whitney between adjacent disorder bins (Bonferroni)")
    by_bin = {b: [num(r["tm_score"]) for r in rows if r["disorder_bin"] == b] for b in BINS}
    pairs = [(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
    testable = [(a, b) for a, b in pairs if len(by_bin[a]) >= 2 and len(by_bin[b]) >= 2]
    k = len(testable)
    emit("  {0} adjacent pair(s) testable (both bins n>=2); Bonferroni factor {0}.".format(k))
    for a, b in pairs:
        na, nb = len(by_bin[a]), len(by_bin[b])
        if (a, b) not in testable:
            emit("  {0} vs {1}: skipped (n={2}, n={3}).".format(a, b, na, nb))
            continue
        u_stat, raw_p = mannwhitneyu(by_bin[a], by_bin[b], alternative="two-sided")
        adj_p = min(raw_p * k, 1.0)
        emit("  {0} vs {1} (n={2}, n={3}): U = {4}, raw p = {5}, Bonferroni p = {6}".format(
            a, b, na, nb, fmt(u_stat), fmt(raw_p), fmt(adj_p)))
    emit("")

    block = "\n".join(lines)
    print(block)

    existing = ""
    if os.path.exists(STATS):
        with open(STATS) as handle:
            existing = handle.read()
    if SENTINEL in existing:
        print("[extra_stats] block already present in {0}; not appending again.".format(STATS))
    else:
        with open(STATS, "a") as handle:
            handle.write(block + "\n")
        print("[extra_stats] appended block to {0}.".format(STATS))


if __name__ == "__main__":
    main()

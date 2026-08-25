#!/usr/bin/env python3
"""compare_metrics.py

Compare viral and cellular proteins on the three headline measures, in one
figure, with a Mann-Whitney U test per measure.

  TM-score        how well the AlphaFold model matches the experimental fold
  mean pLDDT      how confident AlphaFold was
  disorder (%)    how much of the sequence is predicted intrinsically disordered

Each measure gets its own panel and its own axis. Plotting them against a shared
axis would be meaningless, since TM-score runs 0 to 1 while pLDDT runs 0 to 100.

Mann-Whitney is used rather than a t-test because these distributions are skewed
and bounded, and normality is rejected for TM-score in both groups. The
rank-biserial correlation is reported next to each p-value, since with a sample
this size the effect size carries more of the argument than the p-value does.

Reads  ./results/metrics.csv
Writes ./results/compare_metrics.png
       ./results/compare_metrics.txt

Run with:  python src/compare_metrics.py [--all]

By default only coverage-filtered proteins are used, which excludes fragment
mismatches. Pass --all to include every successfully processed protein.
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = "results"

# Categorical palette: identity, two series, fixed order. Validated for
# colour-vision deficiency (worst adjacent pair dE 17.1 protan, 28.6 normal),
# chroma floor and 3:1 contrast against a white surface. Groups are also named
# directly on the x axis, so identity never rests on colour alone.
COLORS = {"viral": "#d1495b", "cellular": "#2077C9"}
GROUPS = ["viral", "cellular"]

# Recessive ink for every piece of text and furniture. Text never wears the
# series colour; the coloured mark beside it carries the identity.
INK = "#1B2430"
INK_MUTED = "#5A6675"
GRID = "#DFE4EA"

PANELS = [
    ("tm_score", "TM-score", "TM-score (AF vs experimental)", None),
    ("mean_plddt", "mean pLDDT", "mean pLDDT", None),
    ("disorder_pct", "predicted disorder", "predicted disorder (%)", None),
]


def load_rows(coverage_filtered=True):
    """Load usable metric rows, optionally dropping fragment mismatches."""
    path = os.path.join(RESULTS, "metrics.csv")
    rows = []
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            if r.get("status") != "ok":
                continue
            if coverage_filtered and r.get("fragment_flag") == "True":
                continue
            rows.append(r)
    return rows


def values(rows, group, field):
    """Numeric values of one field for one group, missing entries dropped."""
    out = []
    for r in rows:
        if r["type"] != group:
            continue
        raw = r.get(field)
        if raw in (None, "", "NA"):
            continue
        out.append(float(raw))
    return np.array(out, dtype=float)


def rank_biserial(u_statistic, n1, n2):
    """Effect size from Mann-Whitney U, in [-1, 1]; positive means viral higher."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    return 2.0 * u_statistic / (n1 * n2) - 1.0


def test_panel(viral, cellular):
    """Return a dict of Mann-Whitney results for one measure."""
    result = {
        "n_viral": len(viral),
        "n_cellular": len(cellular),
        "median_viral": float(np.median(viral)) if len(viral) else float("nan"),
        "median_cellular": float(np.median(cellular)) if len(cellular) else float("nan"),
        "u": None, "p": None, "r": None,
    }
    if len(viral) >= 2 and len(cellular) >= 2:
        u_stat, p_value = mannwhitneyu(viral, cellular, alternative="two-sided")
        result["u"] = float(u_stat)
        result["p"] = float(p_value)
        result["r"] = rank_biserial(u_stat, len(viral), len(cellular))
    return result


def p_label(p_value):
    """Readable significance label for annotating a panel."""
    if p_value is None:
        return "not testable"
    if p_value < 0.001:
        return "p < 0.001"
    return "p = {0:.3f}".format(p_value)


def draw(rows, out_png, subtitle):
    """Draw the three-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
    rng = np.random.default_rng(0)
    stats = {}

    for ax, (field, short, ylabel, _) in zip(axes, PANELS):
        series = [values(rows, g, field) for g in GROUPS]
        stats[field] = test_panel(series[0], series[1])

        # Boxes carry the summary; the jittered points keep every protein
        # visible, which matters at this sample size.
        bp = ax.boxplot(series, positions=[1, 2], widths=.52,
                        showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 1.6},
                        whiskerprops={"color": INK_MUTED, "linewidth": 1},
                        capprops={"color": INK_MUTED, "linewidth": 1})
        for patch, group in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLORS[group])
            patch.set_alpha(.16)
            patch.set_edgecolor(COLORS[group])
            patch.set_linewidth(1.6)

        for pos, group, data in zip([1, 2], GROUPS, series):
            if not len(data):
                continue
            jitter = pos + (rng.random(len(data)) - .5) * .22
            ax.scatter(jitter, data, s=26, color=COLORS[group], alpha=.75,
                       edgecolor="white", linewidth=.6, zorder=3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["viral", "cellular"], color=INK)
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
        ax.set_title(short, color=INK, fontsize=11.5, pad=26, fontweight="semibold")

        # Significance bracket, drawn above the data so it never overlaps it.
        combined = np.concatenate([s for s in series if len(s)]) if any(len(s) for s in series) else np.array([0, 1])
        lo, hi = float(np.min(combined)), float(np.max(combined))
        span = (hi - lo) or 1.0
        bar = hi + span * .10
        ax.plot([1, 1, 2, 2], [bar, bar + span * .035, bar + span * .035, bar],
                color=INK_MUTED, linewidth=1.1, clip_on=False)
        ax.text(1.5, bar + span * .06, p_label(stats[field]["p"]),
                ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_ylim(lo - span * .08, bar + span * .22)

        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9.5)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor=COLORS[g], markeredgecolor="white", label=g)
               for g in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, -.015), fontsize=10, labelcolor=INK)

    fig.suptitle("AlphaFold accuracy, confidence and disorder: viral vs cellular",
                 color=INK, fontsize=13, fontweight="semibold", y=.99)
    fig.text(.5, .925, subtitle, ha="center", color=INK_MUTED, fontsize=9.5)
    fig.tight_layout(rect=[0, .05, 1, .90])
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)
    return stats


def write_report(stats, rows, out_txt, scope):
    """Write the Mann-Whitney results as a readable text report."""
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    n_v = sum(1 for r in rows if r["type"] == "viral")
    n_c = sum(1 for r in rows if r["type"] == "cellular")

    emit("Viral vs cellular: TM-score, pLDDT and predicted disorder")
    emit(scope)
    emit("Mann-Whitney U, two-sided. Rank-biserial r is the effect size;")
    emit("positive r means the viral group ranks higher.")
    emit("")
    emit("Proteins compared: {0} viral, {1} cellular".format(n_v, n_c))
    emit("")

    for field, short, _, _ in PANELS:
        s = stats[field]
        emit(short)
        emit("  median viral    = {0:.4f}  (n={1})".format(s["median_viral"], s["n_viral"]))
        emit("  median cellular = {0:.4f}  (n={1})".format(s["median_cellular"], s["n_cellular"]))
        if s["p"] is None:
            emit("  not enough data in both groups to test.")
        else:
            verdict = "significant" if s["p"] < 0.05 else "not significant"
            emit("  U = {0:.1f}, p = {1:.4g} -> {2} at alpha 0.05".format(
                s["u"], s["p"], verdict))
            emit("  rank-biserial r = {0:+.3f}".format(s["r"]))
        emit("")

    with open(out_txt, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("Wrote {0}".format(out_txt))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--all", action="store_true",
                        help="include fragment-flagged proteins (default excludes them)")
    args = parser.parse_args()

    coverage_filtered = not args.all
    rows = load_rows(coverage_filtered)
    if not rows:
        print("No usable rows in {0}/metrics.csv. Run af_study.py first.".format(RESULTS))
        return

    scope = ("Coverage-filtered set (fragment mismatches excluded)."
             if coverage_filtered else "All successfully processed proteins.")
    os.makedirs(RESULTS, exist_ok=True)
    out_png = os.path.join(RESULTS, "compare_metrics.png")

    stats = draw(rows, out_png, scope)
    print("Wrote {0}".format(out_png))
    write_report(stats, rows, os.path.join(RESULTS, "compare_metrics.txt"), scope)


if __name__ == "__main__":
    main()

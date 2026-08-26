"""Generic bar-chart visualization for any gbm_importances_*.csv table.

All four regression scripts (demo_baseline.py, demo_baseline_filtered.py,
topic_regression_per_review.py, topic_regression_concat_filtered.py) write this same schema:
feature, importance, std, direction, signed_importance, ci_low, ci_high. This
script works unmodified on any of them.

Chart: a horizontal diverging bar chart ("coefficient plot"), one bar per
feature, sorted by unsigned importance descending (most important at top).
--top-n optionally caps how many features are plotted (default: all of them);
pass e.g. --top-n 25 to only show the 25 most important. Bar position/length
is signed_importance, so direction is read directly off which side of x=0 the
bar falls on; bar color also encodes direction (orange = positive/associated
with higher friending bias, blue = negative), reusing the same convention as
the topics_per_class figures. Horizontal error whiskers span [ci_low, ci_high],
putting std's uncertainty on the same axis and in the same units as the bar
itself. This puts all seven columns onto one plot instead of needing separate
panels for magnitude vs. direction vs. uncertainty.

Run from the project root as a module:
    python -m src.eval.plot_gbm_importances outputs/demographics_regression/demo_regression_gbm_importances_filtered.csv
    python -m src.eval.plot_gbm_importances outputs/bertopic/topic_regression/topic_gbm_importances_top85_min_cluster50_concat_filtered.csv
    python -m src.eval.plot_gbm_importances outputs/bertopic/topic_regression/topic_gbm_importances_top235_min_cluster200_per_review.csv --top-n 25
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

POSITIVE_COLOR = "#E69F00"  # same convention as topics_per_class: high_fb / positive
NEGATIVE_COLOR = "#56B4E9"  # low_fb / negative
NEUTRAL_COLOR = "#999999"   # direction == 0 (no correlation signal to sign from)

# Short tokens that actually occur in these tables' feature names / topic
# keywords (grepped from the demo and topic gbm_importances csvs) and should
# stay all-caps rather than being title-cased like an ordinary word.
ACRONYMS = {"ap", "ib", "esl", "iep"}

# Features whose prettified-by-rule name (e.g. "N Words") reads worse than a
# hand-picked one.
FEATURE_LABEL_OVERRIDES = {
    "n_words": "Word Count",
    "n_reviews": "Review Count",
}

# Matches "topic_4 (ap, classes, ap classes)" and captures just the
# parenthesized keywords -- the topic id itself isn't meaningful to a reader.
_TOPIC_RE = re.compile(r"^topic_\d+\s*\((.*)\)\s*$")


def load_importances(csv_path):
    return pd.read_csv(csv_path)


def _prettify_words(text):
    words = text.split()
    return " ".join(w.upper() if w.lower() in ACRONYMS else w.capitalize() for w in words if w)


def _prettify_feature(label):
    """Human-readable version of a raw feature/column name, e.g.
    "students_9_to_12" -> "Students 9 To 12", "topic_4 (ap, classes, ap
    classes)" -> "AP, Classes, AP Classes" -- no topic id, no parentheses, no
    underscores/hyphens left in what actually renders on the chart."""
    label = str(label)
    if label in FEATURE_LABEL_OVERRIDES:
        return FEATURE_LABEL_OVERRIDES[label]
    m = _TOPIC_RE.match(label)
    if m:
        keywords = [kw.strip() for kw in m.group(1).split(",")]
        return ", ".join(_prettify_words(kw) for kw in keywords if kw)
    # Split on underscores/hyphens, then also split each chunk on whitespace
    # (e.g. "ethnicity-Asian or Pacific Islander" already has spaces in its
    # second half) so every individual word gets capitalized, not just the
    # first word of each underscore/hyphen-delimited chunk.
    words = [w for chunk in re.split(r"[_\-]+", label) for w in chunk.split()]
    return " ".join(w.upper() if w.lower() in ACRONYMS else w.capitalize() for w in words if w)


def _bar_color(direction):
    if direction > 0:
        return POSITIVE_COLOR
    if direction < 0:
        return NEGATIVE_COLOR
    return NEUTRAL_COLOR


def plot_importances(df, top_n=None, title=None):
    """Horizontal diverging bar chart, sorted by unsigned importance
    descending, with CI whiskers. Returns the matplotlib Figure. top_n=None
    (the default) plots every row."""
    df = df.sort_values("importance", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    df = df.iloc[::-1].copy()  # largest importance at the top of the plot

    labels = [_prettify_feature(f) for f in df["feature"]]
    signed = df["signed_importance"].to_numpy()
    xerr_low = signed - df["ci_low"].to_numpy()
    xerr_high = df["ci_high"].to_numpy() - signed
    colors = [_bar_color(d) for d in df["direction"]]

    # Left margin scales with the longest label so nothing gets clipped,
    # rather than truncating labels with "...".
    longest_label = max((len(l) for l in labels), default=0)
    fig_width = 9 + 0.09 * longest_label
    fig, ax = plt.subplots(figsize=(fig_width, max(4, 0.35 * len(df))))
    ax.barh(
        labels, signed, color=colors,
        xerr=[xerr_low, xerr_high], capsize=3,
        error_kw={"elinewidth": 1, "alpha": 0.7},
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Signed permutation importance (magnitude x correlation-sign direction)")
    ax.set_title(title or "GBM permutation importance (signed, 95% CI)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=POSITIVE_COLOR, label="positive (higher friending bias)"),
            Patch(facecolor=NEGATIVE_COLOR, label="negative (lower friending bias)"),
        ],
        loc="lower right", fontsize=8,
    )
    fig.tight_layout()
    return fig


def run(csv_path, top_n=None, out_dir=None, title=None):
    # Default to the SAME directory as the input csv: each stage now keeps
    # its tables and figures colocated (outputs/demographics_regression/,
    # outputs/bertopic/topic_regression/, ...) rather than sharing one
    # project-wide outputs/figures/, so a single hardcoded default would be
    # wrong for at least one of the two callers.
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df = load_importances(csv_path)

    fig = plot_importances(df, top_n=top_n, title=title)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = f"{out_dir}/{stem}_signed_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_plotted = len(df) if top_n is None else min(top_n, len(df))
    print(f"Wrote {out_path} ({n_plotted} of {len(df)} features)")
    return out_path


def _parse_top_n(value):
    return None if value.lower() == "all" else int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a gbm_importances_*.csv table")
    parser.add_argument("--top-n", type=_parse_top_n, default=None,
                         help="Number of top features (by unsigned importance) to plot, "
                              "or 'all' (default) to plot every feature")
    parser.add_argument("--out-dir", default=None,
                         help="Defaults to the same directory as csv_path")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    run(args.csv_path, top_n=args.top_n, out_dir=args.out_dir, title=args.title)


if __name__ == "__main__":
    main()

"""
Horizontal diverging bar charts for discovered themes.

On every chart the axis is signed "toward low-FB": RIGHT (+) = more
characteristic of LOW-friending-bias schools, LEFT (-) = more characteristic of
HIGH-FB schools, regardless of how GABRIEL happened to phrase the label.

Adapted from Nabeel's previous code for this pairwise discover/classify setup.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from gabriel.utils.plot_utils import bar_plot

from .themes import clean_theme_label, theme_direction

# bar_plot() gives every row the same height regardless of how many lines its
# wrapped category label needs, so a chart mixing short and long labels ends
# up with the longest ones' text (and its "net %" annotation) overlapping the
# neighboring bars. Sizing the figure off the WORST-CASE wrapped line count
# (rather than bar_plot's own bar-count-only heuristic) fixes that without
# touching the installed package.
_INCHES_PER_LINE = 0.16  # vertical room per wrapped line, at tick_label_size=13
_BASE_ROW_INCHES = 0.10  # minimum room per row even for single-line labels


def _figure_height(labels, wrap_width, min_height=6.0):
    max_lines = max((len(textwrap.wrap(str(l), wrap_width)) or 1 for l in labels), default=1)
    per_row = _BASE_ROW_INCHES + _INCHES_PER_LINE * max_lines
    return max(min_height, per_row * len(labels) + 2)


def _value_axis_limits(values, errors=None, pad_frac=0.3):
    """bar_plot() prints each bar's value as text right past the bar's tip
    (past the error-bar whisker, if any). For the longest bars that tip sits
    right at the plot's own edge, next to zero headroom, so the annotation
    text spills out of the axes and overlaps the category-label margin next
    to it. Padding the axis limits well beyond the widest bar+whisker leaves
    room for that text to render inside the plot instead."""
    errors = errors or [0] * len(values)
    lo = min([v - abs(e) for v, e in zip(values, errors)] + [0])
    hi = max([v + abs(e) for v, e in zip(values, errors)] + [0])
    span = (hi - lo) or 1.0
    return (lo - pad_frac * span, hi + pad_frac * span)


def _theme_ci_halfwidth(clf, orig_label, z=1.96):
    """
    Half-width of a z-based confidence interval for a theme's net_pct, capturing
    sampling noise across schools.

    net_pct = 100 * mean_i(actual_i - inverted_i) over schools i, a paired
    difference (both indicators measured on the same school). Its standard error
    is 100 * std(actual_i - inverted_i) / sqrt(n), so the CI half-width is z*SE.
    `clf` is the per-school classification frame; columns are
    "<orig_label>_actual" / "<orig_label>_inverted". Returns None if unavailable.
    """
    a, v = f"{orig_label}_actual", f"{orig_label}_inverted"
    if clf is None or a not in clf.columns or v not in clf.columns:
        return None

    def _to01(col):
        return clf[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    ai, vi = _to01(a), _to01(v)
    mask = ai.notna() & vi.notna()
    d = (ai[mask] - vi[mask]).astype(float)
    n = len(d)
    if n < 2:
        return None
    se = 100.0 * d.std(ddof=1) / np.sqrt(n)
    return float(z * se)


def plot_discovered_themes(
    summary,
    save_path,
    max_bars=24,
    classification=None,
    ci_z=1.96,
):
    """
    Render a HORIZONTAL bar chart of distinctive themes from a discover `summary`.

    `summary` may be a DataFrame or a path to the saved summary CSV, so the chart
    can be reformatted without re-running the (paid) discover pipeline.

    Each GABRIEL label is a directional claim ("<grp> entries ... more than
    <other> entries") and `net_pct` measures how strongly that claim holds. We
    fold the label's direction into the sign so the plotted value is a single
    "toward low-FB" score:  direction(label) x net_pct.

    If `classification` (the per-school classify frame or its CSV path) is given,
    error bars show a z*SE confidence interval (default 95%) for each bar,
    reflecting sampling noise across schools. NOTE: this captures uncertainty
    from the finite set of schools only, NOT LLM classification noise (raise the
    discover pipeline's n_runs to shrink that).
    """
    if isinstance(summary, (str, Path)):
        summary = pd.read_csv(summary)
    if isinstance(classification, (str, Path)):
        classification = pd.read_csv(classification)

    pct_col = "net_pct" if "net_pct" in summary.columns else "difference_pct"
    df = summary.dropna(subset=[pct_col]).copy()
    df["_orig_label"] = df["label"]
    df["toward_low_fb_pct"] = df.apply(
        lambda r: theme_direction(r["_orig_label"]) * r[pct_col], axis=1
    )
    # CI half-width is computed on the ORIGINAL label (matches classify columns);
    # the sign flip doesn't change the magnitude of the error.
    df["_ci"] = df["_orig_label"].map(
        lambda lab: _theme_ci_halfwidth(classification, lab, ci_z)
    )
    df["label"] = df["label"].map(clean_theme_label)
    df = df.sort_values("toward_low_fb_pct", ascending=False)

    # Keep the most distinctive of each group (top + bottom) if there are many.
    if max_bars and len(df) > max_bars:
        df = pd.concat([df.head((max_bars + 1) // 2), df.tail(max_bars // 2)])

    # Error bars must align with bar order, so we disable bar_plot's own sort.
    error_bars = None
    if df["_ci"].notna().any():
        error_bars = df["_ci"].fillna(0.0).tolist()

    n = len(df)
    wrap_width = 48
    values = df["toward_low_fb_pct"].tolist()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    bar_plot(
        df["label"].tolist(),
        values,
        title="Distinctive review themes: low- vs high-friending-bias schools",
        as_percent=True,
        x_label="Theme",
        y_label="<- more in HIGH-FB schools      net %      more in LOW-FB schools ->",
        orientation="horizontal",
        figsize=(12, _figure_height(df["label"], wrap_width)),
        label_wrap_mode="auto",
        wrap_width=wrap_width,
        label_font_size=11,
        tick_label_size=13,
        horizontal_label_fraction=0.45,
        horizontal_bar_height=0.4,
        max_bars_per_plot=n,
        error_bars=error_bars,
        error_bar_capsize=3,
        value_axis_limits=_value_axis_limits(values, error_bars),
        sort_mode="none",
        save_path=str(save_path),
    )
    print(f"Saved bar chart to {save_path}")
    return df


def plot_prevalence_themes(
    summary,
    save_path,
    max_bars=24,
    title="Theme prevalence: low- vs high-friending-bias schools",
    y_label="<- more prevalent in HIGH-FB     prevalence diff (pts)     more prevalent in LOW-FB ->",
):
    """Horizontal diverging bar chart of theme differences (percentage points)
    with 95% CIs. Positive (right) = more characteristic of low-FB schools."""
    df = summary.dropna(subset=["diff_pct"]).sort_values("diff_pct", ascending=False)
    if max_bars and len(df) > max_bars:
        df = pd.concat([df.head((max_bars + 1) // 2), df.tail(max_bars // 2)])
    err = df["ci_pct"].fillna(0.0).tolist() if "ci_pct" in df.columns else None
    n = len(df)
    wrap_width = 48
    values = df["diff_pct"].tolist()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    bar_plot(
        df["label"].tolist(),
        values,
        title=title,
        as_percent=True,
        x_label="Theme",
        y_label=y_label,
        orientation="horizontal",
        figsize=(12, _figure_height(df["label"], wrap_width)),
        label_wrap_mode="auto",
        wrap_width=wrap_width,
        label_font_size=11,
        tick_label_size=13,
        horizontal_label_fraction=0.45,
        horizontal_bar_height=0.4,
        max_bars_per_plot=n,
        error_bars=err,
        error_bar_capsize=3,
        value_axis_limits=_value_axis_limits(values, err),
        sort_mode="none",
        save_path=str(save_path),
    )
    print(f"Saved prevalence chart to {save_path}")
    return df

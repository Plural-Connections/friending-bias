"""
Data loading and pair construction.

Convention used throughout the package: ``circle`` == low-friending-bias (FB)
schools and ``square`` == high-FB schools (this is how the two corpora are passed
to GABRIEL). Friending-bias estimates come from the Social Capital Atlas column
``bias_own_ses_hs``; a school is "low-FB" if its (mean-centered) estimate is
negative and "high-FB" if positive.

Per-school review text is NOT re-concatenated here the way Nabeel's code did.
It's read straight from ``data/interim/gs_reviews_concat_by_school.csv`` (built by
src/data/concat_reviews.py from the chronologically-sorted per-review file), so
every stage of the project concatenates reviews the same way: oldest-first,
space-joined, no reviewer-role prefix. This makes the reviews pure.
"""

import re

import numpy as np
import pandas as pd


def build_fb_pairs(
    input_file_demos,
    input_file_reviews_concat,
    max_schools_per_group=None,
    seed=None,
    max_words_per_school=None,
):
    """
    Build a DataFrame of (low_fb_reviews, high_fb_reviews) PAIRS — one row per
    pairing of a low-friending-bias school with a high-friending-bias school,
    each school's ALREADY-concatenated reviews (see module docstring). Only
    complete pairs are returned (no padding), so each row is a genuine
    low-vs-high comparison and every school appears in exactly one pair.

    Schools are ranked by extremity so `max_schools_per_group` keeps the most
    extreme of each group. If `seed` is given, each group is then shuffled
    independently before zipping, so the low<->high correspondence within a pair
    is RANDOM rather than rank-matched (extreme-low no longer always paired with
    extreme-high).
    """
    df_demos = pd.read_csv(input_file_demos, dtype={"universal-id": str})[
        ["universal-id", "bias_own_ses_hs"]
    ]
    df_demos = df_demos[~df_demos["bias_own_ses_hs"].isna()].reset_index(drop=True)

    df_reviews = pd.read_csv(input_file_reviews_concat, dtype={"universal-id": str})

    df_school = pd.merge(df_demos, df_reviews, on="universal-id", how="inner")
    df_school = df_school[~df_school["comments"].isna()].reset_index(drop=True)
    df_school = df_school[df_school["comments"].str.len() > 0].reset_index(drop=True)
    df_school = df_school.rename(columns={"comments": "reviews"})

    # ── Group summary: schools / reviews / total words per FB group ─────────
    for _name, _mask in (
        ("low-FB", df_school["bias_own_ses_hs"] < 0),
        ("high-FB", df_school["bias_own_ses_hs"] > 0),
    ):
        _sub = df_school[_mask]
        print(
            f"{_name}: {len(_sub)} schools | "
            f"{int(_sub['n_reviews'].sum())} reviews | {int(_sub['n_words'].sum())} words"
        )

    low = (
        df_school[df_school["bias_own_ses_hs"] < 0]
        .sort_values("bias_own_ses_hs", ascending=True)["reviews"]
        .tolist()
    )
    high = (
        df_school[df_school["bias_own_ses_hs"] > 0]
        .sort_values("bias_own_ses_hs", ascending=False)["reviews"]
        .tolist()
    )
    if max_schools_per_group:
        low = low[:max_schools_per_group]
        high = high[:max_schools_per_group]

    # Equalize text volume per school so the pairwise "which side shows more of
    # theme X" judgment is not confounded by review length (high-FB schools have
    # ~2x longer concatenated reviews, which otherwise wins nearly every
    # comparison regardless of content).
    if max_words_per_school:

        def _cap(text):
            return " ".join(str(text).split()[:max_words_per_school])

        low = [_cap(t) for t in low]
        high = [_cap(t) for t in high]

    # Randomize the low<->high correspondence (shuffle each group independently).
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(low)
        rng.shuffle(high)

    m = min(len(low), len(high))  # complete pairs only
    return pd.DataFrame({"low_fb_reviews": low[:m], "high_fb_reviews": high[:m]})


def _swap_circle_square(text):
    """Swap 'circle' <-> 'square' (case-insensitive) — GABRIEL's inversion."""
    return re.sub(
        r"(?i)circle|square",
        lambda m: "square" if m.group(0).lower() == "circle" else "circle",
        str(text),
    )


def build_combined_labels(base_labels):
    """
    Mirror discover's internal actual/inverted construction: for each base label
    keep it as-is (the 'actual' claim) AND add its circle<->square-swapped twin
    (the 'inverted' claim). Returns (combined_labels, rename_map) where rename_map
    turns the classify output columns into '<label>_actual' / '<label>_inverted'.
    """
    combined, rename = {}, {}
    for lab, desc in base_labels.items():
        swapped = _swap_circle_square(lab)
        if swapped == lab:
            swapped = f"{lab} (inverted)"
        combined[lab] = desc
        rename[lab] = f"{lab}_actual"
        combined[swapped] = _swap_circle_square(desc)
        rename[swapped] = f"{lab}_inverted"
    return combined, rename


def summarize_actual_inverted(clf, base_labels):
    """Build a discover-style summary (label, actual/inverted counts, net_pct)
    from a classify frame whose columns are '<label>_actual'/'<label>_inverted'."""

    def _to01(col):
        return clf[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    recs = []
    for lab in base_labels:
        a, v = f"{lab}_actual", f"{lab}_inverted"
        if a not in clf.columns or v not in clf.columns:
            continue
        ai, vi = _to01(a), _to01(v)
        mask = ai.notna() & vi.notna()
        n = int(mask.sum())
        if n == 0:
            continue
        at, it = float(ai[mask].sum()), float(vi[mask].sum())
        recs.append(
            {
                "label": lab,
                "actual_true": at,
                "inverted_true": it,
                "total": n,
                "actual_pct": 100.0 * at / n,
                "inverted_pct": 100.0 * it / n,
                "net_pct": 100.0 * (at - it) / n,
            }
        )
    return pd.DataFrame(recs)

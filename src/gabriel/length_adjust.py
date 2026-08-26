"""
Length-adjusted pairwise effect sizes.

High-FB schools have ~2x longer concatenated reviews, and the longer side of a
pair tends to "win" nearly every theme comparison regardless of content. These
helpers re-estimate each theme's net difference HOLDING the within-pair review
length gap fixed, so the reported effect reflects content rather than volume.

Adapted from Nabeel's previous code for this pairwise discover/classify setup.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import plot_prevalence_themes
from .themes import clean_theme_label, theme_direction


def length_adjusted_pairwise(clf, z=1.96, n_boot=400, seed=0):
    """
    LENGTH-ADJUSTED version of the PAIRWISE (discover_contrasting_themes_heldout)
    result, computed from its classification frame.

    Per pair i and theme, the paired outcome is an ORDINAL variable
        d_i = actual_i - inverted_i  in {-1, 0, +1}
    (-1 low-FB side won, 0 tie, +1 high-FB side won). The confound is the
    WITHIN-PAIR length gap L_i = log(words_high) - log(words_low): the longer
    side tends to win.

    Because d is ordinal/bounded (not continuous), we model it with an ORDERED
    LOGISTIC (proportional-odds) regression d ~ L rather than OLS, and read the
    length-adjusted net as the expected value at equal length:
        E[d | L=0] = P(+1 | L=0) - P(-1 | L=0)
    The CI is a nonparametric bootstrap over pairs (SE = std of the bootstrap
    E[d|L=0]); if the ordered model can't fit (degenerate categories /
    non-convergence) we fall back to OLS with HC3 heteroskedasticity-robust SEs.
    The estimate is signed toward low-FB via the theme's subject. Positive =>
    more characteristic of low-FB schools at equal review length.
    Columns: label, raw_diff_pct, diff_pct (adjusted, toward-low), ci_pct.
    """
    import statsmodels.formula.api as smf
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    wl = clf["low_fb_reviews"].fillna("").map(lambda t: max(len(str(t).split()), 1))
    wh = clf["high_fb_reviews"].fillna("").map(lambda t: max(len(str(t).split()), 1))
    L = np.log(wh) - np.log(wl)

    def _d01(c):
        return clf[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})

    def _ord_net_at_zero(dd):
        """E[d | L=0] from an ordered-logit fit; None if it cannot fit."""
        y = dd["d"].astype(int)
        if y.nunique() < 2:
            return None
        try:
            m = OrderedModel(y, dd[["L"]], distr="logit").fit(method="bfgs", disp=0)
            probs = m.model.predict(m.params, exog=np.array([[0.0]]))[0]
            return float(np.dot(np.sort(y.unique()), probs))
        except Exception:
            return None

    rng = np.random.default_rng(seed)
    recs = []
    for a in [c for c in clf.columns if c.endswith("_actual")]:
        base = a[: -len("_actual")]
        v = base + "_inverted"
        if v not in clf.columns:
            continue
        d = pd.DataFrame({"d": _d01(a) - _d01(v), "L": L}).dropna()
        if len(d) < 20:
            continue
        raw = 100 * d["d"].mean()

        est = _ord_net_at_zero(d)
        b0 = ci = None
        if est is not None:
            boots = []
            for _ in range(n_boot):
                samp = d.iloc[rng.integers(0, len(d), len(d))]
                e = _ord_net_at_zero(samp)
                if e is not None:
                    boots.append(e)
            if len(boots) >= max(30, n_boot // 2):
                b0 = 100 * est
                ci = z * 100 * float(np.std(boots, ddof=1))
        if b0 is None:  # ordered model unusable -> OLS with robust SEs
            m = smf.ols("d ~ L", data=d).fit(cov_type="HC3")
            b0, ci = 100 * m.params["Intercept"], z * 100 * m.bse["Intercept"]

        dirn = theme_direction(base)
        recs.append(
            {
                "label": clean_theme_label(base),
                "raw_diff_pct": dirn * raw,
                "diff_pct": dirn * b0,
                "ci_pct": ci,
            }
        )
    return pd.DataFrame(recs)


def analyze_heldout_length_adjusted(
    classification_csv,
    save_path,
):
    """
    Post-hoc, NO-API analysis for the PAIRWISE held-out output: re-reads an
    existing classification_heldoutB.csv and reports each theme's net difference
    ADJUSTED for the within-pair review-length gap (see `length_adjusted_pairwise`).
    Bars crossing 0 have no length-adjusted low/high difference.
    """
    clf = pd.read_csv(classification_csv)
    summ = length_adjusted_pairwise(clf)
    csv_path = Path(save_path).with_name(Path(save_path).stem + ".csv")
    summ.to_csv(csv_path, index=False)
    print(f"Saved length-adjusted pairwise summary to {csv_path}")
    if len(summ):
        plot_prevalence_themes(
            summ,
            save_path=save_path,
            title="Pairwise net difference, ADJUSTED for within-pair review length",
            y_label="<- more in HIGH-FB     net diff at equal length (pts)     more in LOW-FB ->",
        )
    else:
        print("No themes to analyze.")
    return summ

"""Binned scatter plots of school-level predictors against friending bias.

Replicates the binned-scatter methodology from the Social Capital Atlas papers
(Chetty et al.): the horizontal-axis variable (each predictor) is split into
ventiles (20 bins of 5 percentile points); within each bin we plot the
student-weighted mean of friending bias against the student-weighted mean of
the predictor. A quadratic best-fit line from weighted OLS is overlaid as a
visual guide to the (possibly non-parametric) relationship. All predictors are
drawn as panels of a single figure and a summary table is written to
outputs/tables/.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

merged_path = "data/interim/gs_demos_with_social_capital.csv"
fig_dir = "outputs/figures"
table_dir = "outputs/tables"

BIAS_COL = "bias_own_ses_hs"  # friending bias by own SES (primary Atlas measure)
# Weight by grades 9-12 enrollment: the population friending bias is defined over.
# (`enrollment` is the whole-school NCES count and is a reasonable alternative.)
WEIGHT_COL = "students_9_to_12"
N_BINS = 20  # ventiles = 5-percentile-point bins

# Ethnicity share columns (percentages, 0-100) used to build the diversity index.
ETHNICITY_COLS = [
    "ethnicity-Hispanic",
    "ethnicity-Black",
    "ethnicity-Two or more races",
    "ethnicity-Asian or Pacific Islander",
    "ethnicity-White",
    "ethnicity-Native Hawaiian or Other Pacific Islander",
    "ethnicity-Native American",
]

HHI_COL = "ethnicity_hhi"  # racial concentration (1 = single group, low = diverse)

# Predictors to plot against friending bias. Add more here as needed.
FACTORS = [
    HHI_COL,
    "ethnicity-White",
    "ethnicity-Black",
    "student-teacher-ratio",
    "exposure_own_ses_hs",
    "bias_parent_ses_hs",
]

LABELS = {
    HHI_COL: "Racial Diversity (HHI)",
    "ethnicity-White": "% White students",
    "ethnicity-Black": "% Black students",
    "student-teacher-ratio": "Student–teacher ratio",
    "exposure_own_ses_hs": "Own-SES exposure",
    "bias_parent_ses_hs": "Friending bias (parent SES)",
}


def add_ethnicity_hhi(df, eth_cols=ETHNICITY_COLS, out_col=HHI_COL):
    """Add a Herfindahl-Hirschman index of racial concentration per school.

    HHI = sum of squared group shares. Shares are the ethnicity percentages
    renormalized to sum to 1 per school (so rows that don't total 100 still
    yield a valid index). Ranges from 1/k (perfectly even mix across k groups)
    to 1 (a single group); higher = less diverse. Rows missing every share are
    left as NaN.
    """
    shares = df[eth_cols].apply(pd.to_numeric, errors="coerce")
    totals = shares.sum(axis=1)
    # Renormalize each row to sum to 1; rows with no data (total 0) -> NaN
    fracs = shares.div(totals.where(totals > 0), axis=0)
    df[out_col] = (fracs ** 2).sum(axis=1, min_count=1)
    return df


def wls_quadratic(x, y, w):
    """Weighted OLS of y on [1, x, x^2]. Returns (coeffs, r2, p_quad).

    coeffs are ordered for np.polyval: [a2, a1, a0] (a2*x^2 + a1*x + a0).
    r2 is the weight-adjusted coefficient of determination; p_quad is the
    two-sided p-value on the quadratic term.
    """
    # Design matrix [1, x, x^2] and a student-weighted quadratic fit
    X = PolynomialFeatures(degree=2).fit_transform(x.reshape(-1, 1))
    model = LinearRegression(fit_intercept=False).fit(X, y, sample_weight=w)
    beta = model.coef_                         # [a0, a1, a2]
    r2 = model.score(X, y, sample_weight=w)    # weighted coefficient of determination

    # sklearn gives no standard errors, so derive the p-value on a2 by hand
    resid = y - model.predict(X)
    n, k = X.shape
    W = np.diag(w)
    cov = (resid @ W @ resid) / (n - k) * np.linalg.inv(X.T @ W @ X)
    se_a2 = np.sqrt(cov[2, 2])
    t_a2 = beta[2] / se_a2 if se_a2 > 0 else np.nan
    p_quad = 2 * stats.t.sf(abs(t_a2), df=n - k) if np.isfinite(t_a2) else np.nan

    return beta[::-1], r2, p_quad  # reverse -> [a2, a1, a0] for np.polyval


def _binned_means(x, y, w, n_bins=N_BINS):
    """Student-weighted mean of x and y within ventiles of x.

    Bin edges are unweighted percentiles of x (standard 'ventile' definition);
    the means *within* each bin are weighted by student counts.
    """
    # qcut on x defines the 5-percentile-point bins; duplicate edges dropped.
    bins = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"x": x, "y": y, "w": w, "bin": bins})
    wmean = lambda t, col: np.average(t[col], weights=t["w"])
    g = d.groupby("bin")
    wx = g.apply(wmean, "x", include_groups=False)
    wy = g.apply(wmean, "y", include_groups=False)
    return wx.to_numpy(), wy.to_numpy()


def plot_factor_panel(ax, df, factor, bias_col=BIAS_COL, weight_col=WEIGHT_COL):
    """Draw one binned-scatter panel (factor -> friending bias). Returns stats."""
    cols = pd.DataFrame({
        "x": pd.to_numeric(df[factor], errors="coerce"),
        "y": pd.to_numeric(df[bias_col], errors="coerce"),
        "w": pd.to_numeric(df[weight_col], errors="coerce"),
    }).dropna()
    cols = cols[cols["w"] > 0]
    x, y, w = cols["x"].to_numpy(), cols["y"].to_numpy(), cols["w"].to_numpy()
    n = len(cols)

    if n < N_BINS:
        ax.set_visible(False)
        print(f"Skipping {factor}: only {n} weighted observations")
        return None

    # Quadratic weighted OLS fit as a visual guide, drawn over the bin range
    coeffs, r2, p_quad = wls_quadratic(x, y, w)

    # Binned scatter points (weighted means within ventiles of the predictor)
    bx, by = _binned_means(x, y, w)
    xs = np.linspace(bx.min(), bx.max(), 200)
    ax.plot(xs, np.polyval(coeffs, xs), linewidth=2, zorder=2)
    ax.scatter(bx, by, s=40, zorder=3)
    ax.axhline(0, linewidth=0.8, linestyle="--", zorder=1)

    label = LABELS.get(factor, factor)
    ax.set_xlabel(label)
    ax.set_ylabel("Friending bias (own SES)")
    ax.set_title(label, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    stats_text = f"$R^2$ = {r2:.3f}\n$p_{{quad}}$ = {p_quad:.0e}\nn = {n:,}"
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    return {
        "predictor": factor,
        "n_schools": n,
        "r_squared": r2,
        "p_quad": p_quad,
        "quad_coef": coeffs[0],   # x^2 term: sign = shape of the curve
    }


def run_correlations(merged_path=merged_path, factors=FACTORS,
                     bias_col=BIAS_COL, out_dir=fig_dir, table_dir=table_dir):
    """Build one combined binned-scatter figure + a summary table."""
    df = pd.read_csv(merged_path)
    add_ethnicity_hhi(df)

    ncols = 3
    nrows = int(np.ceil(len(factors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    results = []
    for ax, factor in zip(axes, factors):
        results.append(plot_factor_panel(ax, df, factor, bias_col))
    for ax in axes[len(factors):]:  # hide unused panels
        ax.set_visible(False)

    fig.suptitle("Predictors of Friending Bias in High Schools Using Own SES",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945,
             "Binned scatter (20 ventiles), student-weighted; "
             "solid line = weighted quadratic fit",
             ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5, w_pad=2.0)
    fig_path = f"{out_dir}/predictors_of_friending_bias.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    summary = pd.DataFrame([r for r in results if r is not None])
    summary = summary.sort_values("r_squared", ascending=False)
    table_path = f"{table_dir}/bias_correlations.csv"
    summary.to_csv(table_path, index=False)
    print(f"Wrote {table_path}\n")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_correlations()

"""Initial EDA plots: friending-bias distribution, reviews-per-school
histogram, and words-per-school histogram.

plot_friending_bias uses the full 13,958-school demos frame (no review
dependency, so filtering doesn't apply). plot_reviews_per_school and
plot_words_per_school use the NCES-FILTERED concat file (not the unfiltered
one) so their reported n and distributions describe the actual ~11,859-school
analytic sample used throughout the rest of the pipeline, rather than the
much larger and mostly-unrelated full ~92,902-school review corpus.

Outputs:
    outputs/demographics_regression/friending_bias_distribution.png
    outputs/demographics_regression/reviews_per_school_hist.png
    outputs/demographics_regression/words_per_school_hist.png
"""

import matplotlib.pyplot as plt
import pandas as pd

merged_path = "data/interim/gs_demos_with_social_capital.csv"
reviews_path = "data/processed/gs_reviews_concat_by_school_nces_filtered.csv"
out_dir = "outputs/demographics_regression"

BIAS_COL = "bias_own_ses_hs"  # friending bias by own SES (primary Atlas measure)


def plot_friending_bias(merged_path, out_path):
    df = pd.read_csv(merged_path)
    bias = df[BIAS_COL].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(bias, bins=60, edgecolor="black", linewidth=0.3)
    ax.axvline(bias.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mean = {bias.mean():.3f}")
    ax.axvline(bias.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"median = {bias.median():.3f}")
    ax.set_xlabel("Friending bias (own SES, high school)")
    ax.set_ylabel("Number of schools")
    ax.set_title(f"Distribution of friending bias (n = {len(bias):,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_reviews_per_school(reviews_path, out_path):
    df = pd.read_csv(reviews_path)
    n = df["n_reviews"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale, clipped so the long tail doesn't flatten the bulk.
    clip = int(n.quantile(0.99))
    ax1.hist(n.clip(upper=clip), bins=range(1, clip + 2),
             color="green", edgecolor="black", linewidth=0.3)
    ax1.set_xlabel(f"Reviews per school (clipped at 99th pct = {clip})")
    ax1.set_ylabel("Number of schools")
    ax1.set_title("Reviews per school (linear, clipped)")

    # Log-scaled x to show the full skew.
    ax2.hist(n, bins=60, color="green", edgecolor="black", linewidth=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Reviews per school (log)")
    ax2.set_ylabel("Number of schools (log)")
    ax2.set_title("Reviews per school (log-log)")

    fig.suptitle(
        f"Reviews per school (n = {len(n):,}; median = {n.median():.0f}, max = {n.max():.0f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_words_per_school(reviews_path, out_path):
    df = pd.read_csv(reviews_path)
    n = df["n_words"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale, clipped so the long tail doesn't flatten the bulk.
    clip = int(n.quantile(0.99))
    ax1.hist(n.clip(upper=clip), bins=60,
             color="purple", edgecolor="black", linewidth=0.3)
    ax1.axvline(n.median(), color="orange", linestyle="--", linewidth=1.5,
                label=f"median = {n.median():.0f}")
    ax1.set_xlabel(f"Words per school (clipped at 99th pct = {clip})")
    ax1.set_ylabel("Number of schools")
    ax1.set_title("Words per school (linear, clipped)")
    ax1.legend()

    # Log-scaled axes to show the full skew.
    ax2.hist(n, bins=60, color="purple", edgecolor="black", linewidth=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Words per school (log)")
    ax2.set_ylabel("Number of schools (log)")
    ax2.set_title("Words per school (log-log)")

    fig.suptitle(
        f"Total review words per school (n = {len(n):,}; median = {n.median():.0f}, max = {n.max():.0f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import os
    os.makedirs(out_dir, exist_ok=True)
    plot_friending_bias(merged_path, f"{out_dir}/friending_bias_distribution.png")
    plot_reviews_per_school(reviews_path, f"{out_dir}/reviews_per_school_hist.png")
    plot_words_per_school(reviews_path, f"{out_dir}/words_per_school_hist.png")

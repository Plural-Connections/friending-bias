"""CONCAT, FILTERED: raw bivariate correlation between each topic share and
friending bias -- a model-free companion to topic_regression_concat_filtered.py.

Where topic_regression_concat_filtered.py fits a GBM on all 85 topic shares
jointly and reports permutation importances, this script looks at each topic
in isolation: the plain Pearson r between that one topic's approximate share
of a school's concatenated reviews and the school's friending bias, with no
model in between. This is the "raw bivariate-correlation check" cited as
corroborating evidence (alongside a synthetic-signal recovery test) for the
claim that the low topic-content R^2 reflects a real ceiling in the data,
not a specification artifact -- see
outputs/archive_metric_tuning_explorations_for_bertopic/ARCHIVE_README.md.

Same data as topic_regression_concat_filtered.py: BERTopic fit with
MIN_CLUSTER_SIZE=50 on the nces-filtered review corpus (85 real topics),
scored on each school's concatenated-review blob, joined on nces_id to
gs_demos_with_social_capital.csv.

Outputs: outputs/bertopic/topic_regression/topic_bivariate_correlations_top{N}_min_cluster{MIN_CLUSTER_SIZE}_concat_filtered.csv
"""

import numpy as np
import pandas as pd
from scipy import stats

MIN_CLUSTER_SIZE = 50
N_TOPICS = 85  # all real topics the MIN_CLUSTER_SIZE=50 filtered refit produced

demos_path = "data/interim/gs_demos_with_social_capital.csv"
topic_distr_path = (
    f"outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_{MIN_CLUSTER_SIZE}"
    f"_min_cluster_by_school_concat_filtered.csv"
)
topic_summary_path = f"outputs/bertopic/unsupervised_exploration/summary/topic_summary_{MIN_CLUSTER_SIZE}_min_cluster_filtered.csv"
table_dir = "outputs/bertopic/topic_regression"
run_tag = f"top{N_TOPICS}_min_cluster{MIN_CLUSTER_SIZE}_concat_filtered"

TARGET = "bias_own_ses_hs"


def load_data(demos_path=demos_path, topic_distr_path=topic_distr_path):
    """Same inner join as topic_regression_concat_filtered.py: only schools
    with both a computed bias score and a topic distribution."""
    demos = pd.read_csv(demos_path, dtype={"nces_id": str})[["nces_id", TARGET]]
    topics = pd.read_csv(topic_distr_path, dtype={"nces_id": str})
    topic_cols = [c for c in topics.columns if c.startswith("topic_")]

    df = demos.merge(topics[["nces_id"] + topic_cols], on="nces_id", how="inner")
    X = df[topic_cols].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].to_numpy()
    return X, y


def topic_feature_labels(topic_cols, path=topic_summary_path):
    """Map each `topic_N` column to its top-3 c-TF-IDF keywords, so the
    output table is legible without cross-referencing topic_summary."""
    keywords = pd.read_csv(path).set_index("topic_id")["keywords"]
    labels = {}
    for col in topic_cols:
        topic_id = int(col.split("_", 1)[1])
        top3 = ", ".join(keywords.loc[topic_id].split(", ")[:3])
        labels[col] = f"{col} ({top3})"
    return labels


def bivariate_correlations(X, y):
    """Plain Pearson r (and its p-value) between each topic column and y,
    one topic at a time -- no model, no other topics controlled for."""
    y = pd.Series(y, index=X.index)
    rows = []
    for col in X.columns:
        x = X[col]
        mask = x.notna() & y.notna()
        r, p = stats.pearsonr(x[mask], y[mask])
        rows.append({"feature": col, "r": r, "abs_r": abs(r), "p_value": p, "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


def run():
    X, y = load_data()
    print(f"n schools = {len(X):,}  |  n topics = {X.shape[1]}\n")

    corrs = bivariate_correlations(X, y)
    labels = topic_feature_labels(X.columns.tolist())
    corrs["feature"] = corrs["feature"].map(labels)

    out_path = f"{table_dir}/topic_bivariate_correlations_{run_tag}.csv"
    corrs.to_csv(out_path, index=False)

    print(f"max |r| = {corrs['abs_r'].max():.4f}  (topic: {corrs.iloc[0]['feature']})")
    print("\n=== Topic bivariate correlations with friending bias (top 10 by |r|) ===")
    print(corrs.head(10).to_string(index=False))
    print(f"\nWrote {out_path}")
    return corrs


if __name__ == "__main__":
    run()

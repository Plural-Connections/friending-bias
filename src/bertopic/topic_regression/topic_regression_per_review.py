"""UNFILTERED, PER_REVIEW: Topic-mixture GBM model for friending bias.

IMPORTANT: this script's counterpart, topic_regression_concat_filtered.py,
is NOT a clean single-variable comparison. The two scripts simultaneously
differ on three axes at once -- (1) filtered vs. unfiltered corpus, (2)
per-review vs. concat topic aggregation, and (3) MIN_CLUSTER_SIZE (200 vs.
50) -- so any R^2 difference between them reflects all three, not filtering
alone. Don't read a headline-number gap between the two as evidence about
filtering in isolation.

Same target, CV scheme, and evaluation/importance machinery as
demo_baseline.py's GBM stage -- but the features here are each school's
approximate topic-mixture distribution over BERTopic topics fitted with
MIN_CLUSTER_SIZE=200 (outputs/bertopic/unsupervised_exploration/
distributions/topic_distributions_top{N}_min_cluster{MIN_CLUSTER_SIZE}
_by_school_per_review.csv), a share per topic computed
per-review and averaged to the school level by
topics_distro_BERT_per_review.py, instead of demographics. This isolates how
much of friending bias the CONTENT of what people say about a school explains.

Two earlier attempts undercounted topics: MIN_CLUSTER_SIZE=50 produced 952
topics with a long fat tail of single-school-name clusters (e.g. "st theresa,
theresa, teresa"), and hand-picking a TOP_N_TOPICS subset (40, then 300) of
those discarded most of a school's real topical mass. Refitting with
MIN_CLUSTER_SIZE=200 collapses that tail down to 235 real topics -- few
enough to use ALL of them as features, no cutoff needed (see
topics_distro_BERT_per_review.py). Every output name embeds MIN_CLUSTER_SIZE
and the topic count so this run is never confused with the earlier two.

Keyed on nces_id (not universal-id): that per-review table is already
restricted to the ~11,859 schools with a matching row in
gs_demos_with_social_capital.csv, so the join here is on nces_id to match it.

Outputs:  outputs/models/topic_regression/topic_gbm_top{N}_min_cluster{MIN_CLUSTER_SIZE}_per_review.pkl
          outputs/bertopic/topic_regression/topic_gbm_metrics_top{N}_min_cluster{MIN_CLUSTER_SIZE}_per_review.csv
          outputs/bertopic/topic_regression/topic_gbm_importances_top{N}_min_cluster{MIN_CLUSTER_SIZE}_per_review.csv
"""

import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from src.eval.metrics import evaluate

MIN_CLUSTER_SIZE = 200
N_TOPICS = 235  # actual real-topic count the MIN_CLUSTER_SIZE=200 refit produced

demos_path = "data/interim/gs_demos_with_social_capital.csv"
topic_distr_path = (
    f"outputs/bertopic/unsupervised_exploration/distributions/topic_distributions_top{N_TOPICS}"
    f"_min_cluster{MIN_CLUSTER_SIZE}_by_school_per_review.csv"
)
topic_summary_path = f"outputs/bertopic/unsupervised_exploration/summary/topic_summary_{MIN_CLUSTER_SIZE}_min_cluster.csv"
model_dir = "outputs/models/topic_regression"
table_dir = "outputs/bertopic/topic_regression"
run_tag = f"top{N_TOPICS}_min_cluster{MIN_CLUSTER_SIZE}_per_review"

TARGET = "bias_own_ses_hs"
N_SPLITS = 5
RANDOM_SEEDS = 0
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEEDS)


def load_data(demos_path=demos_path, topic_distr_path=topic_distr_path):
    """Build X (each school's topic-mixture shares) and y (friending bias).

    Inner join on nces_id: only schools with both a computed bias score
    and a topic distribution (i.e. at least one review) are usable.
    """
    demos = pd.read_csv(demos_path, dtype={"nces_id": str})[["nces_id", TARGET]]
    topics = pd.read_csv(topic_distr_path, dtype={"nces_id": str})
    topic_cols = [c for c in topics.columns if c.startswith("topic_")]

    df = demos.merge(topics[["nces_id"] + topic_cols], on="nces_id", how="inner")
    X = df[topic_cols].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].to_numpy()
    return X, y


def topic_feature_labels(topic_cols, path=topic_summary_path):
    """Map each `topic_N` column to its top-3 c-TF-IDF keywords, so the
    importance table is legible without cross-referencing topic_summary."""
    keywords = pd.read_csv(path).set_index("topic_id")["keywords"]
    labels = {}
    for col in topic_cols:
        topic_id = int(col.split("_", 1)[1])
        top3 = ", ".join(keywords.loc[topic_id].split(", ")[:3])
        labels[col] = f"{col} ({top3})"
    return labels


def fit_gbm_cv(X, y, param_grid=None):
    """Tree model. Try several settings, score each with CV, keep the best.
    Trees ignore feature scale and handle missing values on their own, so no
    impute/scale step is needed here. Same grid as demo_baseline.py."""
    param_grid = param_grid or {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [500, 1000],
    }
    search = GridSearchCV(
        LGBMRegressor(random_state=RANDOM_SEEDS, verbose=-1),
        param_grid,
        scoring="r2",
        cv=CV,
    )
    search.fit(X, y)
    print(f"GBM best params: {search.best_params_}")
    return search.best_estimator_


def evaluate_cv(model, X, y):
    """Honest score. Predict each school with a model trained on the other
    folds, then compare those held-out predictions to the truth."""
    preds = cross_val_predict(clone(model), X, y, cv=CV)
    return evaluate(y, preds)


def gbm_importances(model, X, y, labels=None, confidence=0.95):
    """Rank features by how much the model leans on each. Method is to
    scramble one column at a time and measure how far R^2 falls.

    Permutation importance itself has no sign (shuffling can only hurt or leave
    performance unchanged), so `direction`/`signed_importance` borrow their sign
    from each feature's raw correlation with the target -- a linear/monotonic
    reading of direction, not a claim about the (possibly nonlinear) GBM's exact
    response shape. `ci_low`/`ci_high` are a 95% CI on the mean across the
    n_repeats shuffles (t-distribution, since n_repeats is small), applied to
    the signed value.
    """
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=RANDOM_SEEDS, scoring="r2",
    )
    feature_names = [labels[c] for c in X.columns] if labels else X.columns

    direction = np.sign(X.corrwith(pd.Series(y, index=X.index)).fillna(0).to_numpy())
    signed_importance = result.importances_mean * direction

    n_repeats = result.importances.shape[1]
    se = result.importances_std / np.sqrt(n_repeats)
    margin = stats.t.ppf(1 - (1 - confidence) / 2, df=n_repeats - 1) * se

    return (pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,  # average R^2 drop when shuffled
        "std": result.importances_std,
        "direction": direction,
        "signed_importance": signed_importance,
        "ci_low": signed_importance - margin,
        "ci_high": signed_importance + margin,
    }).sort_values("importance", ascending=False).reset_index(drop=True))


def run():
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    X, y = load_data()
    print(f"n schools = {len(X):,}  |  n features = {X.shape[1]}\n")

    gbm = fit_gbm_cv(X, y)
    metrics = pd.DataFrame([{"model": "gbm", **evaluate_cv(gbm, X, y)}])

    with open(f"{model_dir}/topic_gbm_{run_tag}.pkl", "wb") as f:
        pickle.dump(gbm, f)
    metrics.to_csv(f"{table_dir}/topic_gbm_metrics_{run_tag}.csv", index=False)

    labels = topic_feature_labels(X.columns.tolist())
    importances = gbm_importances(gbm, X, y, labels)
    importances.to_csv(f"{table_dir}/topic_gbm_importances_{run_tag}.csv", index=False)

    print("\n=== Cross-validated metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== GBM permutation importances (topic mixture) ===")
    print(importances.to_string(index=False))
    print(f"\nWrote model to {model_dir}/ and tables to {table_dir}/")
    return metrics, importances


if __name__ == "__main__":
    run()

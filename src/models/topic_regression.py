"""Topic-mixture GBM model for friending bias.

Same target, CV scheme, and evaluation/importance machinery as
demo_baseline.py's GBM stage -- but the features here are each school's
approximate topic-mixture distribution over the top-40 BERTopic topics
(outputs/tables/topic_distributions_top40_by_school.csv), a share per topic
computed by topics_distro_BERT.py, instead of demographics. This isolates how
much of friending bias the CONTENT of what people say about a school explains.

Outputs:  outputs/models/topic_gbm.pkl
          outputs/tables/topic_gbm_metrics.csv
          outputs/tables/topic_gbm_importances.csv
"""

import pickle

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from src.eval.metrics import evaluate

demos_path = "data/interim/gs_demos_with_social_capital.csv"
topic_distr_path = "outputs/tables/topic_distributions_top40_by_school.csv"
topic_summary_path = "outputs/tables/topic_summary_50_min_cluster_top40.csv"
model_dir = "outputs/models"
table_dir = "outputs/tables"

TARGET = "bias_own_ses_hs"
N_SPLITS = 5
RANDOM_SEEDS = 0
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEEDS)


def load_data(demos_path=demos_path, topic_distr_path=topic_distr_path):
    """Build X (each school's topic-mixture shares) and y (friending bias).

    Inner join on universal-id: only schools with both a computed bias score
    and a topic distribution (i.e. at least one review) are usable.
    """
    demos = pd.read_csv(demos_path, dtype={"nces_id": str})[["universal-id", TARGET]]
    topics = pd.read_csv(topic_distr_path)
    topic_cols = [c for c in topics.columns if c.startswith("topic_")]

    df = demos.merge(topics[["universal-id"] + topic_cols], on="universal-id", how="inner")
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


def gbm_importances(model, X, y, labels=None):
    """Rank features by how much the model leans on each. Method is to
    scramble one column at a time and measure how far R^2 falls."""
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=RANDOM_SEEDS, scoring="r2",
    )
    feature_names = [labels[c] for c in X.columns] if labels else X.columns
    return (pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,  # average R^2 drop when shuffled
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True))


def run():
    X, y = load_data()
    print(f"n schools = {len(X):,}  |  n features = {X.shape[1]}\n")

    gbm = fit_gbm_cv(X, y)
    metrics = pd.DataFrame([{"model": "gbm", **evaluate_cv(gbm, X, y)}])

    with open(f"{model_dir}/topic_gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)
    metrics.to_csv(f"{table_dir}/topic_gbm_metrics.csv", index=False)

    labels = topic_feature_labels(X.columns.tolist())
    importances = gbm_importances(gbm, X, y, labels)
    importances.to_csv(f"{table_dir}/topic_gbm_importances.csv", index=False)

    print("\n=== Cross-validated metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== GBM permutation importances (topic mixture) ===")
    print(importances.to_string(index=False))
    print(f"\nWrote model to {model_dir}/ and tables to {table_dir}/")
    return metrics, importances


if __name__ == "__main__":
    run()

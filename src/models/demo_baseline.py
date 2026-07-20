"""Stage 1 — demographics-only baseline for friending bias.

Establishes how much of friending bias (bias_own_ses_hs) school demographics
alone explain, before any text features. This is the comparison point for every
later stage. Two models are fit and compared:

  * ElasticNet  — regularized linear model (captures linear structure)
  * GBM         — HistGradientBoostingRegressor (captures nonlinearity /
                  interactions). Used in place of LightGBM, which is not
                  installed; it plays the same role and handles NaNs natively.

All out-of-sample scores use GroupKFold keyed on NCES district, so schools from
the same district never fall in both train and test (districts share policies /
populations, so a random split would leak and inflate R^2).

Run with:  python -m src.models.demo_baseline
Outputs:   outputs/models/demo_linear.pkl, outputs/models/demo_gbm.pkl
           outputs/tables/stage1_metrics.csv
           outputs/tables/stage1_gbm_importances.csv
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.eval.cv_splits import get_group_kfold
from src.eval.metrics import evaluate

demos_path = "data/interim/gs_demos_with_social_capital.csv"
reviews_path = "data/interim/gs_reviews_concat_by_school.csv"
model_dir = "outputs/models"
table_dir = "outputs/tables"

TARGET = "bias_own_ses_hs"
N_SPLITS = 5

# School demographics only. Deliberately excludes the Atlas social-capital
# measures (ec_*, exposure_*, bias_parent_*, clustering, volunteering): those
# are derived from the same friendship data as the target and would leak.
# `percent-economically-disadvantaged` is dropped (~95% missing).
DEMOGRAPHIC_FEATURES = [
    "student-teacher-ratio",
    "students_9_to_12",  # school size from the Social Capital Atlas (vs GreatSchools `enrollment`)
    "percent-free-and-reduced-price-lunch",
    "percent-students-with-limited-english-proficiency",
    "percentage-of-full-time-teachers-who-are-certified",
    "student-counselor-ratio",
    "percentage-female",
    "percentage-male",
    "ethnicity-Hispanic",
    "ethnicity-Black",
    "ethnicity-Two or more races",
    "ethnicity-Asian or Pacific Islander",
    "ethnicity-White",
    "ethnicity-Native Hawaiian or Other Pacific Islander",
    "ethnicity-Native American",
]

# Review volume / length covariates, credited to Stage 1 so their explanatory
# power is not silently absorbed by later text stages.
REVIEW_FEATURES = ["n_reviews", "n_words", "mean_words_per_review"]

FEATURES = DEMOGRAPHIC_FEATURES + REVIEW_FEATURES


def load_data(demos_path=demos_path, reviews_path=reviews_path):
    """Return (X, y, groups) for schools with a friending-bias target."""
    df = pd.read_csv(demos_path, dtype={"nces_id": str})
    df = df[df[TARGET].notna()].copy()

    # Merge per-school review volume/length; schools with no reviews -> 0.
    rev = pd.read_csv(reviews_path)[["universal-id", "n_reviews", "n_words"]]
    df = df.merge(rev, on="universal-id", how="left")
    df[["n_reviews", "n_words"]] = df[["n_reviews", "n_words"]].fillna(0)
    df["mean_words_per_review"] = np.where(
        df["n_reviews"] > 0, df["n_words"] / df["n_reviews"], 0.0
    )

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].to_numpy()
    groups = df["nces_id"].str[:7]  # NCES district id = first 7 digits
    return X, y, groups


def fit_elasticnet_cv(X, y, groups):
    """Regularized linear baseline. Imputes + scales, then ElasticNetCV self-
    tunes alpha / l1_ratio. Returns a pipeline fit on all rows."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=50,
            cv=5, max_iter=10000, random_state=0)),
    ])
    return pipe.fit(X, y)


def fit_gbm_cv(X, y, groups, param_grid=None):
    """Gradient-boosted trees, tuned with group-aware CV. Handles NaNs
    natively (no imputation/scaling). Returns the best estimator, refit on
    all rows."""
    param_grid = param_grid or {
        "learning_rate": [0.05, 0.1],
        "max_depth": [None, 3],
        "max_leaf_nodes": [31, 63],
    }
    search = GridSearchCV(
        HistGradientBoostingRegressor(random_state=0),
        param_grid,
        scoring="r2",
        cv=get_group_kfold(N_SPLITS),
    )
    search.fit(X, y, groups=groups)
    print(f"GBM best params: {search.best_params_}")
    return search.best_estimator_


def evaluate_cv(model, X, y, groups):
    """Honest out-of-sample R^2 / RMSE via GroupKFold out-of-fold predictions."""
    preds = cross_val_predict(
        clone(model), X, y,
        groups=groups, cv=get_group_kfold(N_SPLITS),
    )
    return evaluate(y, preds)


def gbm_importances(model, X, y):
    """Permutation feature importances (HistGBR has no native importances)."""
    result = permutation_importance(
        model, X, y, n_repeats=5, random_state=0, scoring="r2",
    )
    return (pd.DataFrame({
        "feature": X.columns,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True))


def run():
    X, y, groups = load_data()
    print(f"n schools = {len(X):,}  |  n districts = {groups.nunique():,}  "
          f"|  n features = {X.shape[1]}\n")

    linear = fit_elasticnet_cv(X, y, groups)
    gbm = fit_gbm_cv(X, y, groups)

    metrics = pd.DataFrame([
        {"model": "elasticnet", **evaluate_cv(linear, X, y, groups)},
        {"model": "gbm", **evaluate_cv(gbm, X, y, groups)},
    ])

    # Persist artifacts
    with open(f"{model_dir}/demo_linear.pkl", "wb") as f:
        pickle.dump(linear, f)
    with open(f"{model_dir}/demo_gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)
    metrics.to_csv(f"{table_dir}/stage1_metrics.csv", index=False)

    importances = gbm_importances(gbm, X, y)
    importances.to_csv(f"{table_dir}/stage1_gbm_importances.csv", index=False)

    print("\n=== Cross-validated metrics (GroupKFold by district) ===")
    print(metrics.to_string(index=False))
    print("\n=== GBM permutation importances (known confounders) ===")
    print(importances.to_string(index=False))
    print(f"\nWrote models to {model_dir}/ and tables to {table_dir}/")
    return metrics, importances


if __name__ == "__main__":
    run()

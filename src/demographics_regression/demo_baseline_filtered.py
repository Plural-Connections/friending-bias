"""FILTERED: Same as demo_baseline.py, but using the nces-filtered concatenated
reviews file (data/processed/gs_reviews_concat_by_school_nces_filtered.csv) for
the review-volume features, instead of the unfiltered
gs_reviews_concat_by_school.csv.

NOTE: because that filtered file was itself built via an INNER join on
nces_id upstream (concat_reviews_nces_filtered.py), it only has rows for
schools whose nces_id matched -- so a school with real reviews but a failed
nces_id match gets n_reviews/n_words=0 here (this script still left-joins +
fillna(0) onto the demos frame, same as demo_baseline.py; only the review
source differs), understating its true review volume. See demo_baseline.py's
docstring for why that script deliberately reads the unfiltered file instead.

Outputs:   outputs/models/demographics_regression/demo_linear_filtered.pkl, outputs/models/demographics_regression/demo_gbm_filtered.pkl
           outputs/demographics_regression/demo_regression_metrics_filtered.csv
           outputs/demographics_regression/demo_regression_gbm_importances_filtered.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from src.eval.metrics import evaluate

demos_path = "data/interim/gs_demos_with_social_capital.csv"
reviews_path = "data/processed/gs_reviews_concat_by_school_nces_filtered.csv"
model_dir = "outputs/models/demographics_regression"
table_dir = "outputs/demographics_regression"

TARGET = "bias_own_ses_hs"
N_SPLITS = 5
RANDOM_SEEDS = 0
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEEDS)

# School demographics only. `percent-economically-disadvantaged` is dropped (~95% missing).
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

# Review volume / length features so their explanatory power is not silently absorbed by later text stages.
REVIEW_FEATURES = ["n_reviews", "n_words", "mean_words_per_review"]

FEATURES = DEMOGRAPHIC_FEATURES + REVIEW_FEATURES


def load_data(demos_path=demos_path, reviews_path=reviews_path):
    """Build X (the features) and y (the value we predict)."""
    df = pd.read_csv(demos_path, dtype={"nces_id": str})

    # Attach how many reviews / words each school has. Left join keeps every
    # school; a school with no reviews gets 0 (a real count, not "missing").
    rev = pd.read_csv(reviews_path)[["universal-id", "n_reviews", "n_words"]]
    df = df.merge(rev, on="universal-id", how="left")
    df[["n_reviews", "n_words"]] = df[["n_reviews", "n_words"]].fillna(0)
    # Average review length, guarding against divide-by-zero for review-less schools.
    df["mean_words_per_review"] = np.where(
        df["n_reviews"] > 0, df["n_words"] / df["n_reviews"], 0.0
    )

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")  # feature columns, forced numeric
    y = df[TARGET].to_numpy()
    return X, y


def fit_elasticnet_cv(X, y):
    """Linear model. Fill the gaps, put every feature on the same scale, then fit a regularized linear
    regression that auto-tunes how hard it shrinks its coefficients."""
    pipe = Pipeline([
        # Fill the only column with gaps (limited-English %) using its median.
        ("impute", SimpleImputer(strategy="median")),
        # Rescale features to mean 0 / sd 1 so the penalty judges them fairly
        # regardless of units (school size in thousands vs. shares in percent).
        ("scale", StandardScaler()),
        # Linear regression + regularization (shrinks coefficients toward 0 to
        # avoid overfitting); the CV picks the best shrinkage strength & style.
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=50,
            cv=CV, max_iter=10000, random_state=RANDOM_SEEDS)),
    ])
    return pipe.fit(X, y)


def fit_gbm_cv(X, y, param_grid=None):
    """Tree model. Try several settings, score each with CV, keep the best.
    Trees ignore feature scale and handle missing values on their own, so no
    impute/scale step is needed here."""
    param_grid = param_grid or {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [500, 1000],
    }
    # GridSearchCV trains every combination and keeps the best
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
    folds, then compare those held-out predictions to the truth. This is
    out-of-sample performance, not memorized training fit."""
    preds = cross_val_predict(
        clone(model), X, y, cv=CV,  # clone an untrained copy for each fold
    )
    return evaluate(y, preds)  # R^2 / RMSE on predictions the model never trained on


def gbm_importances(model, X, y, confidence=0.95):
    """Rank features by how much the model leans on each. Method is to scramble one
    column at a time and measure how far R^2 falls.

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
    direction = np.sign(X.corrwith(pd.Series(y, index=X.index)).fillna(0).to_numpy())
    signed_importance = result.importances_mean * direction

    n_repeats = result.importances.shape[1]
    se = result.importances_std / np.sqrt(n_repeats)
    margin = stats.t.ppf(1 - (1 - confidence) / 2, df=n_repeats - 1) * se

    return (pd.DataFrame({
        "feature": X.columns,
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

    linear = fit_elasticnet_cv(X, y)
    gbm = fit_gbm_cv(X, y)

    metrics = pd.DataFrame([
        {"model": "elasticnet", **evaluate_cv(linear, X, y)},
        {"model": "gbm", **evaluate_cv(gbm, X, y)},
    ])

    # Persist artifacts
    with open(f"{model_dir}/demo_linear_filtered.pkl", "wb") as f:
        pickle.dump(linear, f)
    with open(f"{model_dir}/demo_gbm_filtered.pkl", "wb") as f:
        pickle.dump(gbm, f)
    metrics.to_csv(f"{table_dir}/demo_regression_metrics_filtered.csv", index=False)

    importances = gbm_importances(gbm, X, y)
    importances.to_csv(f"{table_dir}/demo_regression_gbm_importances_filtered.csv", index=False)

    print("\n=== Cross-validated metrics ===")
    print(metrics.to_string(index=False))
    print("\n=== GBM permutation importances (known confounders) ===")
    print(importances.to_string(index=False))
    print(f"\nWrote models to {model_dir}/ and tables to {table_dir}/")
    return metrics, importances


if __name__ == "__main__":
    run()

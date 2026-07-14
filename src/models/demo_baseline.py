"""Stage 1 — demographics-only baseline. Comparison point for everything downstream."""
from src.eval.cv_splits import get_group_kfold
from src.eval.metrics import evaluate

def fit_elasticnet_cv(X, y, groups):
    raise NotImplementedError

def fit_gbm_cv(X, y, groups, param_grid):
    raise NotImplementedError

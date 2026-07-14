"""Stage 5 — text-only concept regression (no demographics)."""

def fit_elasticnet(X_concepts, y, groups):
    raise NotImplementedError

def fit_gbm_shap(X_concepts, y, groups):
    raise NotImplementedError

def bootstrap_coefs(X, y, groups, n_boot: int = 1000):
    """Cluster-aware bootstrap: resample districts, not rows."""
    raise NotImplementedError

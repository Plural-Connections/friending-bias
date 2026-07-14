"""Stage 6 — residualized model (text controlling for demographics). Likely headline result."""

def get_residuals(demo_model, X_demo, y):
    return y - demo_model.predict(X_demo)

def fit_on_residuals(X_concepts, residuals, groups):
    raise NotImplementedError

"""Shared evaluation metrics — import everywhere rather than reimplementing per stage."""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

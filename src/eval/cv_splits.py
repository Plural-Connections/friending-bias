"""Shared cross-validation splitters.

Import this everywhere a model is trained on school-level data — never use
plain KFold, since schools within the same district/state are not
independent observations.
"""
from sklearn.model_selection import GroupKFold


def get_group_kfold(n_splits: int = 5):
    """Returns a GroupKFold instance. Pass groups=df['district_id'] (or
    'state' for a coarser split) when calling .split()/.get_n_splits().
    """
    return GroupKFold(n_splits=n_splits)

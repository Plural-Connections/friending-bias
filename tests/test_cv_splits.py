from src.eval.cv_splits import get_group_kfold

def test_group_kfold_instantiates():
    gkf = get_group_kfold(n_splits=5)
    assert gkf.get_n_splits() == 5

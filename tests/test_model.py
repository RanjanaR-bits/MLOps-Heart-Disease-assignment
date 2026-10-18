from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import RAW_FEATURES, TARGET
from src.train import candidates, holdout_metrics, tune_and_evaluate


def test_candidates_include_two_models():
    assert {"logistic_regression", "random_forest"} <= set(candidates())


def test_tune_and_evaluate_returns_valid_metrics(clean_df):
    X, y = clean_df[RAW_FEATURES], clean_df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=1)
    res = tune_and_evaluate("lr", LogisticRegression(max_iter=500),
                            {"model__C": [0.1, 1]}, X_tr, y_tr, X_te, y_te)
    for key in ("test_accuracy", "test_precision", "test_recall", "test_roc_auc"):
        assert 0.0 <= res["test"][key] <= 1.0
    assert res["test"]["test_roc_auc"] > 0.6  # synthetic signal is learnable
    assert "cv_roc_auc_mean" in res["cv"]


def test_holdout_metrics_perfect_case():
    import numpy as np
    m = holdout_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0

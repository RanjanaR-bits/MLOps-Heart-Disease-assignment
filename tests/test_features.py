import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.config import RAW_FEATURES, TARGET
from src.features import add_features, build_pipeline


def test_add_features_values():
    df = pd.DataFrame({"age": [60], "thalach": [160], "trestbps": [150]})
    out = add_features(df)
    assert out.loc[0, "hr_ratio"] == 1.0
    assert out.loc[0, "high_bp"] == 1
    assert "hr_ratio" not in df.columns  # input not mutated


def test_pipeline_handles_missing_and_unseen(clean_df):
    X, y = clean_df[RAW_FEATURES], clean_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    new = X.head(3).copy()
    new.loc[new.index[0], "ca"] = np.nan
    new.loc[new.index[1], "thal"] = 99.0  # unseen category must not crash
    proba = pipe.predict_proba(new)
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1)

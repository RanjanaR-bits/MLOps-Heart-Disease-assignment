import numpy as np
import pandas as pd
import pytest


def make_synthetic(n=240, seed=0):
    """Small fake dataset with the same schema as the cleaned UCI file."""
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n) + target * 4,
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.normal(130, 15, n).round() + target * 6,
        "chol": rng.normal(245, 45, n).round(),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.normal(150, 20, n).round() - target * 15,
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.gamma(1.5, 0.7, n) + target * 0.8, 1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": target,
    })
    df.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return df


@pytest.fixture()
def clean_df():
    return make_synthetic()

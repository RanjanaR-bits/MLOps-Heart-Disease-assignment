"""Feature engineering + preprocessing, packaged as one sklearn Pipeline.

Because the engineered features live INSIDE the pipeline, the API only needs to
send raw patient fields and training/serving can never drift apart.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ENGINEERED_NUMERIC = ["hr_ratio"]
ENGINEERED_FLAGS = ["high_bp"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add two domain features.

    hr_ratio : max heart rate achieved / age-predicted max (220 - age)
    high_bp  : resting blood pressure >= 140 mmHg (hypertension flag)
    """
    out = df.copy()
    out["hr_ratio"] = out["thalach"] / (220 - out["age"])
    out["high_bp"] = (out["trestbps"] >= 140).astype(int)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + ENGINEERED_NUMERIC),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("flags", "passthrough", ENGINEERED_FLAGS),
    ])


def build_pipeline(estimator) -> Pipeline:
    """raw columns -> engineered features -> impute/scale/encode -> classifier."""
    return Pipeline([
        ("engineer", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])

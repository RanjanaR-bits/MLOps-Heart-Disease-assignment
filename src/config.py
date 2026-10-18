"""Central place for paths, column names and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "heart-disease-risk"

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

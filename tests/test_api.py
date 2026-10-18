import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.config import RAW_FEATURES, TARGET
from src.features import build_pipeline

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


@pytest.fixture()
def client(tmp_path, monkeypatch, clean_df):
    model = build_pipeline(LogisticRegression(max_iter=500))
    model.fit(clean_df[RAW_FEATURES], clean_df[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_ok(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk_probability"] <= 1.0


def test_predict_with_optional_fields_missing(client):
    payload = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_bad_input(client):
    assert client.post("/predict", json={**SAMPLE, "age": -5}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "sex": 3}).status_code == 422
    missing = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=missing).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "api_predictions_total" in text

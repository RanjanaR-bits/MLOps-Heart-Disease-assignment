"""Tune, evaluate and track two classifiers with MLflow; save the best for serving.

Usage:  python -m src.train
View runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)

from src.config import (EXPERIMENT_NAME, FIG_DIR, METRICS_PATH, MODEL_PATH, RANDOM_STATE,
                        REPORT_DIR, TEST_SIZE)
from src.data import load_clean
from src.features import build_pipeline

plt.switch_backend("Agg")

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def candidates():
    """Model name -> (estimator, hyper-parameter grid). Grid keys target the 'model' step."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000),
            {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }


def holdout_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def tune_and_evaluate(name, estimator, grid, X_train, y_train, X_test, y_test):
    """Grid-search (5-fold, ROC-AUC), then report CV + hold-out metrics for the best model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(estimator), grid, scoring="roc_auc", cv=cv,
                          n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    scores = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
    cv_metrics = {}
    for m in SCORING:
        cv_metrics[f"cv_{m}_mean"] = float(scores[f"test_{m}"].mean())
        cv_metrics[f"cv_{m}_std"] = float(scores[f"test_{m}"].std())

    proba = best.predict_proba(X_test)[:, 1]
    test_metrics = {f"test_{k}": float(v) for k, v in holdout_metrics(y_test, proba).items()}
    return {
        "name": name, "model": best, "best_params": search.best_params_,
        "cv": cv_metrics, "test": test_metrics, "proba": proba,
    }


def save_plots(name, y_test, proba):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = FIG_DIR / f"confusion_{name}.png"
    roc_path = FIG_DIR / f"roc_{name}.png"

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= 0.5).astype(int), display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC curve - {name}")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    return [cm_path, roc_path]


def save_combined_roc(results, y_test):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r in results:
        RocCurveDisplay.from_predictions(y_test, r["proba"], name=r["name"], ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC comparison (hold-out set)")
    fig.tight_layout()
    path = FIG_DIR / "roc_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_metrics_table(results, winner):
    lines = ["| Model | CV ROC-AUC | CV Acc | CV Prec | CV Recall | Test ROC-AUC | Test Acc "
             "| Test Prec | Test Recall |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c, t = r["cv"], r["test"]
        star = " (selected)" if r["name"] == winner else ""
        lines.append(
            f"| {r['name']}{star} | {c['cv_roc_auc_mean']:.3f} ± {c['cv_roc_auc_std']:.3f} "
            f"| {c['cv_accuracy_mean']:.3f} | {c['cv_precision_mean']:.3f} "
            f"| {c['cv_recall_mean']:.3f} | {t['test_roc_auc']:.3f} | {t['test_accuracy']:.3f} "
            f"| {t['test_precision']:.3f} | {t['test_recall']:.3f} |")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics_table.md").write_text("\n".join(lines) + "\n")


def main():
    X, y = load_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    for name, (estimator, grid) in candidates().items():
        res = tune_and_evaluate(name, estimator, grid, X_train, y_train, X_test, y_test)
        plots = save_plots(name, y_test, res["proba"])
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**res["best_params"], "model_type": name,
                               "n_train": len(X_train), "n_test": len(X_test),
                               "test_size": TEST_SIZE, "seed": RANDOM_STATE, "cv_folds": 5})
            mlflow.log_metrics({**res["cv"], **res["test"]})
            for p in plots:
                mlflow.log_artifact(str(p), artifact_path="plots")
            mlflow.sklearn.log_model(
                res["model"], artifact_path="model",
                signature=infer_signature(X_train, res["model"].predict(X_train)),
                input_example=X_train.head(3))
            res["run_id"] = run.info.run_id
        print(f"{name}: CV AUC={res['cv']['cv_roc_auc_mean']:.3f} "
              f"test AUC={res['test']['test_roc_auc']:.3f} params={res['best_params']}")
        results.append(res)

    winner = max(results, key=lambda r: r["cv"]["cv_roc_auc_mean"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["model"], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "selected_model": winner["name"], "best_params": winner["best_params"],
        **winner["cv"], **winner["test"]}, indent=2, default=str))

    combined = save_combined_roc(results, y_test)
    write_metrics_table(results, winner["name"])
    MlflowClient().set_tag(winner["run_id"], "selected_for_serving", "true")
    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="serving")
        mlflow.log_artifact(str(combined), artifact_path="plots")
    print(f"Selected '{winner['name']}' -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


MODEL_PATH = "models/model.pkl"
EVAL_PATH = "data/eval.csv"
METRICS_PATH = "outputs/metrics.json"
EVAL_THRESHOLD = 0.60


def evaluate(
    model_path: str = MODEL_PATH,
    eval_path: str = EVAL_PATH,
    metrics_path: str = METRICS_PATH,
) -> dict:
    """
    Đánh giá model đã train trên tập eval và lưu metrics ra file JSON.
    Script này được dùng cho DVC stage `evaluate`.
    """
    model = joblib.load(model_path)
    df_eval = pd.read_csv(eval_path)

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    preds = model.predict(X_eval)

    accuracy = float(accuracy_score(y_eval, preds))
    f1 = float(f1_score(y_eval, preds, average="weighted", zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "eval_threshold": EVAL_THRESHOLD,
        "passed": accuracy >= EVAL_THRESHOLD,
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    print(f"Eval gate passed: {metrics['passed']}")

    return metrics


if __name__ == "__main__":
    evaluate()
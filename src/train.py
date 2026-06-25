import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def _normalize_params(params: dict) -> dict:
    """
    Chuẩn hóa params đọc từ YAML.
    Nếu max_depth được ghi là chuỗi "None" thì đổi thành Python None.
    """
    params = dict(params)

    if params.get("max_depth") in ["None", "none", "null", ""]:
        params["max_depth"] = None

    return params


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa siêu tham số cho RandomForestClassifier.
        data_path: đường dẫn đến file train.
        eval_path: đường dẫn đến file eval.

    Trả về:
        accuracy trên tập eval.
    """
    params = _normalize_params(params)

    # 1. Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tách feature và label
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # 3. Cấu hình MLflow local nếu user chưa set tracking URI
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("wine-quality-random-forest")

    with mlflow.start_run():
        # 4. Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(df_train))
        mlflow.log_param("eval_rows", len(df_eval))

        # 5. Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 6. Evaluate
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted", zero_division=0))

        # 7. Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 8. Lưu metrics cho CI/CD
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "eval_threshold": EVAL_THRESHOLD,
        }

        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # 9. Lưu model cho deploy
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # 10. Log artifacts/model vào MLflow
        mlflow.log_artifact("outputs/metrics.json")
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    return acc


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train(params)
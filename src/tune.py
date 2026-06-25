import json
import os
from itertools import product

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = "data/train_phase1.csv"
EVAL_PATH = "data/eval.csv"

EXPERIMENT_NAME = "wine-quality-hyperparameter-tuning"


def build_param_grid(grid: dict) -> list[dict]:
    """
    Chuyển một grid dạng dict thành list các tổ hợp params.
    Ví dụ:
        {"a": [1, 2], "b": [3, 4]}
    thành:
        [{"a": 1, "b": 3}, {"a": 1, "b": 4}, ...]
    """
    keys = list(grid.keys())
    values = [grid[key] for key in keys]

    configs = []
    for combo in product(*values):
        configs.append(dict(zip(keys, combo)))

    return configs


def make_model(algorithm: str, params: dict):
    """
    Khởi tạo model theo tên algorithm.

    LogisticRegression dùng StandardScaler vì model tuyến tính nhạy với scale.
    Tree-based models như RandomForest và GradientBoosting không cần scaling.
    """
    if algorithm == "random_forest":
        return RandomForestClassifier(**params, random_state=42)

    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(**params, random_state=42)

    if algorithm == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**params, random_state=42, max_iter=1000),
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main() -> None:
    """
    Chạy hyperparameter tuning cho nhiều algorithms và log kết quả vào MLflow.

    Nếu MLFLOW_TRACKING_URI đã được set bởi GitHub Actions/DagsHub thì log remote.
    Nếu chưa có, script sẽ log vào MLflow local sqlite.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = pd.read_csv(TRAIN_PATH)
    df_eval = pd.read_csv(EVAL_PATH)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    search_space = {
        "random_forest": build_param_grid(
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5],
            }
        ),
        "gradient_boosting": build_param_grid(
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.10],
                "max_depth": [2, 3],
            }
        ),
        "logistic_regression": build_param_grid(
            {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            }
        ),
    }

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    results = []
    best_result = None
    best_model = None

    for algorithm, configs in search_space.items():
        for idx, params in enumerate(configs, start=1):
            run_name = f"{algorithm}_{idx}"

            with mlflow.start_run(run_name=run_name):
                model = make_model(algorithm, params)
                model.fit(X_train, y_train)

                preds = model.predict(X_eval)
                accuracy = float(accuracy_score(y_eval, preds))
                f1 = float(f1_score(y_eval, preds, average="weighted", zero_division=0))

                result = {
                    "algorithm": algorithm,
                    "run_name": run_name,
                    "accuracy": accuracy,
                    "f1_score": f1,
                    **params,
                }

                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("run_name", run_name)
                mlflow.log_param("train_rows", len(df_train))
                mlflow.log_param("eval_rows", len(df_eval))
                mlflow.log_params(params)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)

                results.append(result)

                if best_result is None or accuracy > best_result["accuracy"]:
                    best_result = result
                    best_model = model

                print(
                    f"{run_name}: accuracy={accuracy:.4f}, "
                    f"f1_score={f1:.4f}, params={params}"
                )

    results_df = pd.DataFrame(results).sort_values(
        by=["accuracy", "f1_score"], ascending=False
    )

    results_df.to_csv("outputs/tuning_results.csv", index=False)

    with open("outputs/best_tuning_params.json", "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    joblib.dump(best_model, "models/best_tuned_model.pkl")

    with mlflow.start_run(run_name="best_tuned_model_summary"):
        mlflow.log_params(
            {
                key: value
                for key, value in best_result.items()
                if key not in ["accuracy", "f1_score"]
            }
        )
        mlflow.log_metric("accuracy", best_result["accuracy"])
        mlflow.log_metric("f1_score", best_result["f1_score"])
        mlflow.log_artifact("outputs/tuning_results.csv")
        mlflow.log_artifact("outputs/best_tuning_params.json")
        mlflow.sklearn.log_model(best_model, "model")

    print("\nBest tuning result:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()
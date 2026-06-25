import json
import os

import numpy as np
import pandas as pd

from src.train import train


FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Wine Quality để dùng trong unit test.
    Dataset này được tạo trong thư mục tạm của pytest, nên không phụ thuộc vào DVC hoặc cloud storage.
    """
    rng = np.random.default_rng(0)
    n = 200

    # Tạo ma trận feature ngẫu nhiên có cùng số cột với dữ liệu thật.
    X = rng.random((n, len(FEATURE_NAMES)))

    # Tạo nhãn ngẫu nhiên cho 3 class: 0, 1, 2.
    y = rng.integers(0, 3, size=n)

    # Tạo DataFrame đúng schema mà src.train.train() mong đợi.
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")

    # Chia dữ liệu giả thành train/eval.
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """
    Kiểm tra hàm train() trả về accuracy dạng float và nằm trong khoảng hợp lệ.
    """
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """
    Kiểm tra file outputs/metrics.json được tạo và có đủ metrics chính.
    """
    train_path, eval_path = _make_temp_data(tmp_path)

    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")

    with open("outputs/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_model_file_created(tmp_path):
    """
    Kiểm tra file models/model.pkl được tạo sau khi training.
    """
    train_path, eval_path = _make_temp_data(tmp_path)

    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")
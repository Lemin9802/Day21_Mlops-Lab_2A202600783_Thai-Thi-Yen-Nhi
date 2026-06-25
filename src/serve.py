import os

import joblib
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel


GCS_BUCKET = os.environ.get("GCS_BUCKET", "day21-mlops-2a202600783-forward-alchemy")
GCS_MODEL_KEY = os.environ.get("GCS_MODEL_KEY", "models/latest/model.pkl")
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

LABEL_MAP = {
    0: "thap",
    1: "trung_binh",
    2: "cao",
}


def download_model() -> None:
    """
    Tải file model.pkl từ GCS về VM khi server khởi động.

    Trên VM, service dùng Google Cloud default credentials.
    Trên GitHub Actions hoặc local, có thể dùng GOOGLE_APPLICATION_CREDENTIALS nếu cần.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)

    if not blob.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model trên GCS: gs://{GCS_BUCKET}/{GCS_MODEL_KEY}"
        )

    blob.download_to_filename(MODEL_PATH)
    print(f"Model đã được tải từ gs://{GCS_BUCKET}/{GCS_MODEL_KEY} về {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Wine Quality MLOps Service")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra trạng thái serving service.
    """
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "gcs_model": f"gs://{GCS_BUCKET}/{GCS_MODEL_KEY}",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra: JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features, received {len(req.features)}",
        )

    pred = int(model.predict([req.features])[0])
    label = LABEL_MAP.get(pred, "unknown")

    return {
        "prediction": pred,
        "label": label,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
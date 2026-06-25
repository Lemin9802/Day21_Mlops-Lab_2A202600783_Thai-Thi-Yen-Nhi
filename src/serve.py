from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc.
    """
    # TODO 1: Tao storage.Client()
    # client = storage.Client()

    # TODO 2: Lay bucket va blob tuong ung
    # bucket = client.bucket(GCS_BUCKET)
    # blob = bucket.blob(GCS_MODEL_KEY)

    # TODO 3: Tai file model xuong may
    # blob.download_to_filename(MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    # print("Model da duoc tai xuong tu GCS.")

    pass  # xoa dong nay sau khi hoan thanh tat ca TODO ben tren


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    pass  # xoa dong nay sau khi hoan thanh


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    # TODO 6: Kiem tra so luong dac trung.
    # Neu len(req.features) != 12, raise HTTPException(status_code=400, ...)

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    # pred = model.predict(...)

    # TODO 8: Tra ve dict chua "prediction" va "label".
    # Nhan tuong ung: 0 -> "thap", 1 -> "trung_binh", 2 -> "cao"

    pass  # xoa dong nay sau khi hoan thanh tat ca TODO ben tren


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

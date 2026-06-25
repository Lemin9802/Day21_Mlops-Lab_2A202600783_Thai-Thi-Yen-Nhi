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
    Tải file model.pkl tu GCS về máy khi server khởi động.

    Hàm này được gọi một lần khi module được import. Sử dụng
    GOOGLE_APPLICATION_CREDENTIALS để xác thực.
    """
    # TODO 1: Tạo storage.Client()
    # client = storage.Client()

    # TODO 2: Lấy bucket và blob tương ứng
    # bucket = client.bucket(GCS_BUCKET)
    # blob = bucket.blob(GCS_MODEL_KEY)

    # TODO 3: Tải file model xuống máy
    # blob.download_to_filename(MODEL_PATH)

    # TODO 4: In thông báo thành công
    # print("Model đã được tải xuống từ GCS.")

    pass  # xóa dòng này sau khi hoàn thành tất cả TODO bên trên


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra sức khỏe server.
    """
    # TODO 5: Trả về dict {"status": "ok"}
    pass  # xóa dòng này sau khi hoàn thành


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Đầu vào : JSON {"features": [f1, f2, ..., f12]}
    Đầu ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    # TODO 6: Kiểm tra số lượng đặc trưng.
    # Nếu len(req.features) != 12, raise HTTPException(status_code=400, ...)

    # TODO 7: Gọi model.predict([req.features]) để lấy kết quả dự đoán.
    # pred = model.predict(...)

    # TODO 8: Trả về dict chứa "prediction" va "label".
    # Nhãn tương ứng: 0 -> "thap", 1 -> "trung_binh", 2 -> "cao"

    pass  # xóa dòng này sau khi hoàn thành tất cả TODO ben tren


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

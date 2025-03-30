from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from model import train_hmm, detect_anomaly

app = FastAPI()

class SensorData(BaseModel):
    Timestamp: str
    Temperature: float
    Pressure: float
    Vibration: float
    Cycle_Count: int

@app.post("/predict")
async def predict(data: List[SensorData]):
    data_dicts = [d.dict() for d in data]
    model, X = train_hmm(data_dicts)
    idx, score = detect_anomaly(model, X)

    if idx is not None:
        return {
            "anomaly_detected": True,
            "anomaly_time": data_dicts[idx]["Timestamp"],
            "anomaly_index": idx,
            "anomaly_score": round(score, 2)
        }
    else:
        return {
            "anomaly_detected": False,
            "message": "No anomaly detected"
        }
@app.get("/")
def read_root():
    return {"message": "FastAPI HMM API is running"}

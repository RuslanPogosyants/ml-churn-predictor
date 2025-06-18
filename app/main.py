from fastapi import FastAPI
from app.schemas import CustomerData
from app.model import load_model, predict

app = FastAPI()
model = load_model()


@app.get("/")
def read_root():
    return {"message": "ML Churn Prediction API"}


@app.post("/predict")
def get_prediction(data: CustomerData):
    result = predict(model, data)
    return {"churn_probability": result}

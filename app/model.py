import os
import joblib
import pandas as pd
from app.utils import preprocess_input


def load_model():
    path = os.path.join(os.path.dirname(__file__), "..", "model/model", "churn_model.pkl")
    path = os.path.abspath(path)
    return joblib.load(path)


def predict(model, data):
    df = preprocess_input(data.dict())
    result = model.predict_proba(df)[0][1]
    return round(result, 3)

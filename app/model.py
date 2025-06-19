import os
import joblib
import pandas as pd
from app.utils import preprocess_input


def load_model():
    base = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(base, "..", "model", "model", "churn_model.pkl"))
    return joblib.load(model_path)


def predict(model, data):
    df = preprocess_input(data.dict())
    result = model.predict_proba(df)[0][1]
    return round(result, 3)

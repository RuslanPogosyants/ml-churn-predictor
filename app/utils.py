import pandas as pd
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "model", "expected_columns.txt"))
with open(path, "r") as f:
    EXPECTED_COLUMNS = [line.strip() for line in f]

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in EXPECTED_COLUMNS:
        if col not in df:
            df[col] = 0

    return df[EXPECTED_COLUMNS]

import pandas as pd


def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    expected_columns = [
        'age',
        'income',
        'gender_female',
        'gender_male',
        'contract_type_month-to-month',
        'contract_type_one-year',
        'contract_type_two-year',
    ]

    for col in expected_columns:
        if col not in df:
            df[col] = 0

    return df[expected_columns]

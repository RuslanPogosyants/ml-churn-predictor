
# ML Churn Predictor

Predict customer churn using an ML model wrapped in FastAPI + Docker + CI.

## Стек
- Python 3.11, Pandas, Scikit‑Learn
- FastAPI, Uvicorn
- Docker & docker-compose
- Pytest, GitHub Actions (CI)

## Структура
```

```text
ml-churn-predictor/
├── app/
├── model/
├── data/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/
```

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python model/train.py
uvicorn app.main:app --reload
```

## В Docker

```bash
docker-compose up --build
```

Будет доступен по: [http://localhost:8000/docs](http://localhost:8000/docs).

## Тестирование

```bash
pytest -q
```

## Использование API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 960
    }'
```

## Модель

* Алгоритм: RandomForestClassifier
* Метрика: see classification report in console


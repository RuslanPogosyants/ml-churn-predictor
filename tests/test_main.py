from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "age": 30,
        "gender": "male",
        "income": 40000,
        "contract_type": "month-to-month"
    })
    assert response.status_code == 200
    assert "churn_probability" in response.json()

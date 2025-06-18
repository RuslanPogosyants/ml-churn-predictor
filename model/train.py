import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

data = pd.DataFrame({
    "age": [25, 45, 35, 52, 23],
    "gender": ["male", "female", "female", "male", "male"],
    "income": [30000, 54000, 42000, 63000, 27000],
    "contract_type": ["month-to-month", "two-year", "one-year", "month-to-month", "two-year"],
    "churn": [0, 1, 0, 1, 0]
})

data = pd.get_dummies(data, columns=["gender", "contract_type"])
X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/churn_model.pkl")

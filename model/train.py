import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

df = pd.read_csv("./data/telco_churn_sample.csv")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf, f"{model_dir}/churn_model.pkl")

with open(f"{model_dir}/expected_columns.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

DATA_PATH = Path("data/data.csv")
METRICS_DIR = Path("metrics")


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    df = df.drop("customerID", axis=1)

    # TotalCharges contains whitespace for missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Encode target: Yes -> 1, No -> 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Label-encode remaining categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # Threshold at 0.5 for binary predictions
    y_pred = (model.predict(X_test_s) >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }


def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }


def main():
    METRICS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_preprocess()

    lr_metrics = train_linear_regression(X_train, X_test, y_train, y_test)
    xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)

    with open(METRICS_DIR / "linear_regression.json", "w") as f:
        json.dump(lr_metrics, f, indent=2)

    with open(METRICS_DIR / "xgboost.json", "w") as f:
        json.dump(xgb_metrics, f, indent=2)

    print(f"Linear Regression — accuracy: {lr_metrics['accuracy']}, f1: {lr_metrics['f1_score']}")
    print(f"XGBoost           — accuracy: {xgb_metrics['accuracy']}, f1: {xgb_metrics['f1_score']}")


if __name__ == "__main__":
    main()

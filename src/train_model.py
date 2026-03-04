import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_PATH = os.path.join("model", "fraud_model.pkl")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Put creditcard.csv in /data")

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' not found in dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("Rows:", len(df))
    print("Fraud count (Class=1):", int((y == 1).sum()))
    print("Non-fraud count (Class=0):", int((y == 0).sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
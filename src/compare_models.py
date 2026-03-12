import os
import json
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def safe_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # fallback simplu
        import numpy as np
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def evaluate_model(model, X_test, y_test):
    y_prob = safe_proba(model, X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
    }


def main(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Place creditcard.csv in that path and run again."
        )

    df = pd.read_csv(data_path)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "model": model_name,
            **metrics
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["pr_auc", "roc_auc", "f1_score"],
        ascending=False
    )

    os.makedirs("model", exist_ok=True)

    results_json = {
        "dataset": os.path.basename(data_path),
        "test_size": 0.2,
        "random_state": 42,
        "ranking_metric": "pr_auc",
        "results": results_df.to_dict(orient="records"),
    }

    with open("model/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    results_df.to_csv("model/model_comparison.csv", index=False)

    print("\nModel comparison saved:")
    print("- model/model_comparison.json")
    print("- model/model_comparison.csv")
    print("\nRanking:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/creditcard.csv",
        help="Path to Kaggle creditcard.csv dataset"
    )
    args = parser.parse_args()
    main(args.data)
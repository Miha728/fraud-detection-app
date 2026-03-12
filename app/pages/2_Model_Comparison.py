import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="centered")

st.title("📊 Model Comparison")
st.caption("Comparison between Logistic Regression, Decision Tree and Random Forest on the Credit Card Fraud Detection dataset.")

comparison_path = "model/model_comparison.json"

if not os.path.exists(comparison_path):
    st.warning(
        "Comparison file not found.\n\n"
        "Run this locally first:\n"
        "`python src/compare_models.py --data data/creditcard.csv`\n\n"
        "Then commit `model/model_comparison.json` and `model/model_comparison.csv`."
    )
    st.stop()

with open(comparison_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

results = payload.get("results", [])
if not results:
    st.error("Comparison file exists, but contains no results.")
    st.stop()

df = pd.DataFrame(results)

st.subheader("Summary")
col1, col2, col3 = st.columns(3)

best_model = df.iloc[0]["model"]
best_pr_auc = df.iloc[0]["pr_auc"]
best_roc_auc = df.iloc[0]["roc_auc"]

col1.metric("Best model", best_model)
col2.metric("Best PR-AUC", f"{best_pr_auc:.4f}")
col3.metric("Best ROC-AUC", f"{best_roc_auc:.4f}")

st.subheader("Detailed results")
st.dataframe(df, use_container_width=True)

st.subheader("PR-AUC by model")
chart_df = df.set_index("model")[["pr_auc"]]
st.bar_chart(chart_df)

st.subheader("ROC-AUC by model")
chart_df_roc = df.set_index("model")[["roc_auc"]]
st.bar_chart(chart_df_roc)

st.subheader("Interpretation")
st.markdown(
    f"""
- **Best model by PR-AUC:** `{best_model}`
- **Why PR-AUC matters here:** fraud detection is a strongly imbalanced classification problem, so PR-AUC is usually more informative than raw accuracy.
- **Current live app remains untouched:** the production demo still uses the existing model from `model/fraud_model.pkl`.
"""
)
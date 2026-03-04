import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import sklearn

st.write("sklearn:", sklearn.__version__)

st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="centered")


def load_model():
    return joblib.load("model/fraud_model.pkl")


def hhmm_to_seconds(time_str: str) -> float:
    """
    Convert "9:15 AM" or "21:30" -> seconds from 00:00.
    Note: creditcard.csv 'Time' is seconds since first transaction,
    but for demo we map time-of-day to seconds-of-day.
    """
    time_str = time_str.strip()

   
    for fmt in ("%I:%M %p", "%I %p"):
        try:
            dt = datetime.strptime(time_str.upper(), fmt)
            return float(dt.hour * 3600 + dt.minute * 60)
        except ValueError:
            pass

    
    for fmt in ("%H:%M", "%H"):
        try:
            dt = datetime.strptime(time_str, fmt)
            return float(dt.hour * 3600 + dt.minute * 60)
        except ValueError:
            pass

    raise ValueError("Time format invalid. Exemple: '9:15 AM', '2 PM', '21:30', '23'")


def build_model_input(model, amount_value: float, time_seconds: float) -> pd.DataFrame:
    """
    Build a 1-row DataFrame that matches EXACTLY the feature names & order
    used during model training.
    """
    if not hasattr(model, "feature_names_in_"):
        features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    else:
        features = list(model.feature_names_in_)

    row = {col: 0.0 for col in features}

    
    if "Time" in row:
        row["Time"] = float(time_seconds)
    if "Amount" in row:
        row["Amount"] = float(amount_value)

   
    df = pd.DataFrame([row], columns=features) 
    return df


def ml_predict_proba(model, df: pd.DataFrame) -> float:
    """
    Return probability of fraud class (1) if available, else use decision_function -> sigmoid-ish.
    For LogisticRegression, predict_proba exists.
    """
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(df)[0][1])
   
    if hasattr(model, "decision_function"):
        import math
        score = float(model.decision_function(df)[0])
        return 1.0 / (1.0 + math.exp(-score))
    
    return float(model.predict(df)[0])


def apply_risk_rules(
    base_prob: float,
    time_seconds: float,
    country: str,
    tx_last_24h: int,
    is_online: bool,
    new_device: bool,
) -> tuple[float, str, list[str], int]:
    """
    Adds a rule-based score on top of the ML probability.
    Returns: final_prob_clamped, risk_level, reasons, rule_points
    """

    reasons = []
    rule_points = 0

    if time_seconds >= 22 * 3600:
        rule_points += 20
        reasons.append("Late-night transaction (after 22:00) +20")

    if is_online:
        rule_points += 15
        reasons.append("Online transaction +15")

    
    if tx_last_24h >= 6:
        rule_points += 25
        reasons.append("High frequency (≥6 transactions in last 24h) +25")
    elif tx_last_24h >= 3:
        rule_points += 10
        reasons.append("Moderate frequency (3–5 transactions in last 24h) +10")

    
    if new_device:
        rule_points += 20
        reasons.append("New device used +20")

    
    if country not in ("Romania", "Other"):
        rule_points += 5
        reasons.append("Non-default country selected +5")

    
    uplift = min(0.35, (rule_points / 100.0) * 0.35)
    final_prob = base_prob + uplift
    final_prob = max(0.0, min(1.0, final_prob))

    if final_prob >= 0.65:
        level = "HIGH"
    elif final_prob >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"

    return final_prob, level, reasons, rule_points



st.title("💳 Fraud Detection App")
st.caption("Demo: ML probability (creditcard-style features) + Risk Rules (operational signals).")

model = load_model()

with st.expander("🔧 Demo notes (important)"):
    st.write(
        "- Modelul ML este antrenat pe un set de date în care feature-urile sunt `Time`, `Amount` și `V1..V28` (anonimizate).\n"
        "- Câmpurile din interfață (țară, device nou, online, frecvență) NU există în datasetul original, deci sunt folosite ca *Risk Rules*.\n"
        "- Rezultatul afișează atât probabilitatea ML, cât și scorul final ajustat + explicația (reason codes)."
    )


colA, colB = st.columns(2)
with colA:
    if st.button("🟢 Preset: Normal"):
        st.session_state["amount"] = 25.0
        st.session_state["time_str"] = "1:15 PM"
        st.session_state["country"] = "Romania"
        st.session_state["tx24"] = 1
        st.session_state["online"] = False
        st.session_state["new_device"] = False

with colB:
    if st.button("🔴 Preset: High-risk"):
        st.session_state["amount"] = 1200.0
        st.session_state["time_str"] = "11:45 PM"
        st.session_state["country"] = "Other"
        st.session_state["tx24"] = 8
        st.session_state["online"] = True
        st.session_state["new_device"] = True


amount_value = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=float(st.session_state.get("amount", 100.0)),
    step=1.0,
)

time_str = st.text_input(
    "Transaction Time (ex: 9:15 AM sau 21:30)",
    value=str(st.session_state.get("time_str", "2:30 PM")),
)

country = st.selectbox(
    "Country",
    options=["Romania", "Other", "Germany", "UK", "USA"],
    index=["Romania", "Other", "Germany", "UK", "USA"].index(st.session_state.get("country", "Romania")),
)

tx_last_24h = st.slider(
    "Transactions in last 24h",
    min_value=0,
    max_value=20,
    value=int(st.session_state.get("tx24", 1)),
)

is_online = st.checkbox("Online transaction", value=bool(st.session_state.get("online", False)))
new_device = st.checkbox("New device used", value=bool(st.session_state.get("new_device", False)))

threshold = st.slider("Alert threshold (final score)", 0.05, 0.95, 0.35, 0.05)

st.divider()

if st.button("Check Transaction"):
    try:
        time_seconds = hhmm_to_seconds(time_str)
        df = build_model_input(model, amount_value, time_seconds)

        base_prob = ml_predict_proba(model, df)
        final_prob, level, reasons, rule_points = apply_risk_rules(
            base_prob=base_prob,
            time_seconds=time_seconds,
            country=country,
            tx_last_24h=tx_last_24h,
            is_online=is_online,
            new_device=new_device,
        )

        # Display results
        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("ML fraud probability", f"{base_prob*100:.1f}%")
        c2.metric("Rule points", f"+{rule_points}")
        c3.metric("Final risk score", f"{final_prob*100:.1f}%")

        if final_prob >= threshold:
            st.error(f"⚠ HIGH RISK (level: {level}) — Flagged for review")
        else:
            st.success(f"✅ OK (level: {level}) — Not flagged")

        st.subheader("Why this result")
        if reasons:
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.write("- No rule-based risk signals triggered.")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")


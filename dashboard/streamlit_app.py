"""
Streamlit dashboard for UPI Fraud Detector.
Pages: Overview, Analytics, Prediction.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# allow importing from src/ and app/ when running via `streamlit run`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_artifact, load_risk_thresholds
from src.explainability import get_explainer, explain_single_prediction, get_top_reasons
from app.schemas import TransactionRequest
from app.features import build_feature_row, scale_feature_row
from app.risk_scoring import get_risk_assessment


st.set_page_config(page_title="UPI Fraud Detector", page_icon="🔍", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/fraud_dataset.csv")
    return df


@st.cache_resource
def load_model_artifacts():
    try:
        model = load_artifact("artifacts/final_model.joblib")
        scaler = load_artifact("artifacts/preprocessor.joblib")
        freq_maps = load_artifact("artifacts/freq_maps.joblib")
        thresholds = load_risk_thresholds("artifacts/risk_thresholds.json")
        explainer = get_explainer(model)
        return model, scaler, freq_maps, thresholds, explainer
    except FileNotFoundError as e:
        st.error(f"Required model artifact not found: {e}. Please ensure the model has been trained and artifacts are present in the `artifacts/` directory.")
        st.stop()


df = load_data()
model, scaler, freq_maps, risk_thresholds, explainer = load_model_artifacts()

st.sidebar.title("UPI Fraud Detector")
st.sidebar.caption("Fraud detection & risk scoring system")
page = st.sidebar.radio("Navigate", ["Overview", "Analytics", "Prediction"])


# Overview page — high-level fraud statistics
if page == "Overview":
    st.title("UPI Fraud Detector — Overview")

    total_transactions = len(df)
    fraud_transactions = df['is_fraud'].sum()
    fraud_rate = (fraud_transactions / total_transactions) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.markdown("---")
    st.subheader("Fraud vs Genuine Distribution")

    chart_col, _ = st.columns([1, 2])
    with chart_col:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['is_fraud'].value_counts().rename({0: "Genuine", 1: "Fraud"}).plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel("Count")
        st.pyplot(fig)


# Analytics page — fraud patterns by amount, time, and receiver history
elif page == "Analytics":
    st.title("UPI Fraud Detector — Analytics")

    st.subheader("Transaction Amount by Fraud Status")
    amt_col, _ = st.columns([1, 2])
    with amt_col:
        df_labeled = df.copy()
        df_labeled['fraud_label'] = df_labeled['is_fraud'].map({0: "Genuine", 1: "Fraud"})

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        df_labeled.boxplot(column='amount', by='fraud_label', ax=ax1)
        ax1.set_xlabel("")
        ax1.set_ylabel("Amount")
        plt.suptitle("")
        ax1.set_title("")
        st.pyplot(fig1)

    st.markdown("---")
    st.subheader("Fraud Rate by Hour of Day")
    time_col, _ = st.columns([1, 2])
    with time_col:
        fraud_by_hour = df.groupby('transaction_time_of_day')['is_fraud'].mean() * 100
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fraud_by_hour.plot(kind='bar', ax=ax2, color='#3498db')
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Fraud Rate (%)")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Fraud Rate by Receiver Transaction History (Binned)")
    hist_col, _ = st.columns([1, 2])
    with hist_col:
        bins = pd.qcut(df['receiver_transaction_history'], 5, duplicates='drop')
        fraud_by_history = df.groupby(bins)['is_fraud'].mean() * 100

        clean_labels = [f"{int(max(interval.left, 0))}–{int(interval.right)}" for interval in fraud_by_history.index]

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.bar(clean_labels, fraud_by_history.values, color='#9b59b6')
        ax3.set_xlabel("Receiver Transaction History (Bucket)")
        ax3.set_ylabel("Fraud Rate (%)")
        st.pyplot(fig3)


# Prediction page — live fraud risk scoring with SHAP explanation
elif page == "Prediction":
    st.title("UPI Fraud Detector — Predict Transaction Risk")
    st.caption("Fill in all fields below, then click Predict to generate a result.")

    with st.form("prediction_form"):
        user_id = st.text_input("User ID", value="U12345")

        st.subheader("Transaction Details")

        transaction_fields = [
            ("amount", "Amount (₹)", dict(min_value=1.0, max_value=50000.0, value=3500.0, help="Transaction amount in INR")),
            ("session_duration", "Session Duration (sec)", dict(min_value=0, max_value=1800, value=120, help="App session length before this transaction")),
            ("receiver_transaction_history", "Receiver Transaction History", dict(min_value=0, max_value=200, value=20, help="Past transactions completed by the receiver")),
            ("transaction_amount_vs_sender_history", "Amount vs Sender History (ratio)", dict(min_value=0.0, max_value=10.0, value=1.0, help="This amount vs sender's typical size (1.0 = average)")),
            ("geographic_disparity", "Geographic Disparity", dict(min_value=0.0, max_value=5.0, value=0.1, help="Mismatch between usual and current location")),
            ("time_between_link_click_and_transaction", "Time Between Link Click and Transaction (sec)", dict(min_value=0, max_value=600, value=30)),
            ("request_amount_roundness", "Request Amount Roundness", dict(min_value=0.0, max_value=1.0, value=0.1, help="How close the amount is to a round number")),
            ("time_to_respond_to_request", "Time to Respond to Request (sec)", dict(min_value=0, max_value=300, value=5)),
        ]

        transaction_values = {}
        cols = st.columns(2)
        for i, (key, label, kwargs) in enumerate(transaction_fields):
            with cols[i % 2]:
                transaction_values[key] = st.number_input(label, **kwargs)

        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            transaction_time_of_day = st.slider("Transaction Hour", 0, 23, 14)
        with slider_col2:
            request_acceptance_rate = st.slider(
                "Request Acceptance Rate", 0.0, 1.0, 0.9,
                help="How often this sender accepts incoming payment requests"
            )

        st.subheader("Device & Behavioral Signals")

        behavioral_fields = [
            ("input_timing_consistency", "Input Timing Consistency", dict(min_value=0.0, max_value=1.0, value=0.8, help="Consistency of typing/input timing vs usual pattern")),
            ("keyboard_input_speed", "Keyboard Input Speed", dict(min_value=0.0, max_value=2.0, value=0.5, help="Relative typing speed during the session")),
            ("input_pause_patterns", "Input Pause Patterns", dict(min_value=0.0, max_value=1.0, value=0.1, help="Irregularity in pauses between inputs")),
            ("screen_active_time", "Screen Active Time (sec)", dict(min_value=0, max_value=1800, value=90)),
            ("geographic_location_vs_ip", "Geographic Location vs IP", dict(min_value=0.0, max_value=5.0, value=0.05, help="Mismatch between GPS and IP-based location")),
            ("background_data_usage", "Background Data Usage", dict(min_value=0.0, max_value=5.0, value=0.3, help="Data usage by background apps during the session")),
            ("pin_entry_speed", "PIN Entry Speed", dict(min_value=0.0, max_value=2.0, value=0.4, help="Relative speed at which the UPI PIN was entered")),
        ]

        behavioral_values = {}
        cols2 = st.columns(2)
        for i, (key, label, kwargs) in enumerate(behavioral_fields):
            with cols2[i % 2]:
                behavioral_values[key] = st.number_input(label, **kwargs)

        submitted = st.form_submit_button("Predict")

    if submitted:
        request_data = TransactionRequest(
            user_id=user_id,
            amount=transaction_values["amount"],
            session_duration=transaction_values["session_duration"],
            receiver_transaction_history=transaction_values["receiver_transaction_history"],
            transaction_amount_vs_sender_history=transaction_values["transaction_amount_vs_sender_history"],
            geographic_disparity=transaction_values["geographic_disparity"],
            transaction_time_of_day=transaction_time_of_day,
            time_between_link_click_and_transaction=transaction_values["time_between_link_click_and_transaction"],
            input_timing_consistency=behavioral_values["input_timing_consistency"],
            keyboard_input_speed=behavioral_values["keyboard_input_speed"],
            input_pause_patterns=behavioral_values["input_pause_patterns"],
            screen_active_time=behavioral_values["screen_active_time"],
            geographic_location_vs_ip=behavioral_values["geographic_location_vs_ip"],
            background_data_usage=behavioral_values["background_data_usage"],
            pin_entry_speed=behavioral_values["pin_entry_speed"],
            request_amount_roundness=transaction_values["request_amount_roundness"],
            request_acceptance_rate=request_acceptance_rate,
            time_to_respond_to_request=transaction_values["time_to_respond_to_request"]
        )

        raw_row = build_feature_row(request_data, freq_maps)
        scaled_row = scale_feature_row(raw_row, scaler)

        fraud_probability = float(model.predict_proba(scaled_row)[:, 1][0])
        is_fraud = int(fraud_probability >= 0.5)
        risk = get_risk_assessment(fraud_probability, risk_thresholds)

        st.markdown("---")
        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Prediction", "Fraud" if is_fraud else "Genuine")
        result_col2.metric("Fraud Probability", f"{fraud_probability:.2%}")
        result_col3.metric("Risk Score", f"{risk['risk_score']}/100", risk['risk_category'])

        progress_col, _ = st.columns([1, 1])
        with progress_col:
            st.progress(risk['risk_score'] / 100)

        ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")
        st.caption(f"Prediction generated at {ist_time}")

        st.markdown("---")
        st.subheader("Why this prediction?")

        explanation = explain_single_prediction(explainer, scaled_row)
        top_reasons = get_top_reasons(explanation, top_n=3)

        reason_labels = [feat for feat, _, _ in top_reasons]
        reason_values = [shap_val for _, shap_val, _ in top_reasons]
        reason_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in reason_values]

        explain_col, _ = st.columns([1, 1])
        with explain_col:
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            ax4.barh(reason_labels, reason_values, color=reason_colors)
            ax4.axvline(0, color='black', linewidth=0.8)
            ax4.set_xlabel("Impact on Fraud Risk (SHAP value)")
            ax4.invert_yaxis()
            st.pyplot(fig4)

        for feature, shap_val, feat_val in top_reasons:
            direction = "increased" if shap_val > 0 else "decreased"
            st.write(f"**{feature}** = {feat_val:.2f} → {direction} fraud risk (impact: {shap_val:+.3f})")

        st.divider()
        st.caption(
            "Disclaimer: This output is generated by a model trained on synthetic data for "
            "educational/portfolio purposes only. It does not represent real fraud detection accuracy."
        )
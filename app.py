"""
Job & Company Trust Scoring System - Streamlit Application
A proof-of-concept ML tool to evaluate trust indicators for hiring companies.
"""

import pickle
from pathlib import Path

import streamlit as st

# Configuration
MODEL_PATH = Path(__file__).parent / "scam_detector.pkl"


def load_model():
    """Load the trained model from disk."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_risk(model, features: list) -> tuple[int, float]:
    """
    Predict class and probability for high-risk.
    Returns (predicted_class, risk_probability).
    """
    import numpy as np

    X = np.array([features])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    # Index 1 = high risk class
    risk_prob = float(proba[1])
    return int(pred), risk_prob


def risk_score_to_label(risk_percent: float) -> str:
    """Convert risk percentage to Low / Medium / High Risk label."""
    if risk_percent < 35:
        return "Low Risk"
    elif risk_percent < 70:
        return "Medium Risk"
    else:
        return "High Risk"


def main():
    st.set_page_config(
        page_title="Job & Company Trust Scoring",
        page_icon="🛡️",
        layout="centered",
    )

    # Header
    st.title("🛡️ Job & Company Trust Scoring System")
    st.caption("Proof of Concept — Trust & Safety Research")

    st.markdown(
        """
        This tool helps evaluate **Trust Level** based on common risk indicators 
        associated with job postings. It is intended for research and demonstration purposes.
        """
    )

    # Disclaimer
    with st.expander("⚠️ Disclaimer", expanded=True):
        st.markdown(
            """
            **This is a demo ML prototype** and should not be used for production decisions.
            Results are based on a limited set of features and sample data. 
            Always conduct your own due diligence when evaluating job opportunities.
            """
        )

    # Load model
    if not MODEL_PATH.exists():
        st.error(
            "Model not found. Please run `python train_model.py` first to train the model."
        )
        st.stop()

    model = load_model()

    # Input section
    st.subheader("Risk Indicators")
    st.markdown("Enter the following information about the job posting or company:")

    col1, col2 = st.columns(2)

    with col1:
        has_website = st.selectbox(
            "Has a company website?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=1,
        )
        uses_company_email = st.selectbox(
            "Uses company email domain (e.g., @company.com)?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=1,
        )
        employee_count = st.number_input(
            "Employee count (approximate)",
            min_value=1,
            max_value=100000,
            value=50,
            step=1,
        )

    with col2:
        asks_money = st.selectbox(
            "Asks for money or payment upfront?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=0,
        )
        has_reviews = st.selectbox(
            "Has verified reviews or ratings?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            index=1,
        )

    # Predict
    if st.button("Evaluate Trust Level", type="primary"):
        features = [
            has_website,
            uses_company_email,
            employee_count,
            asks_money,
            has_reviews,
        ]
        pred_class, risk_prob = predict_risk(model, features)
        risk_percent = risk_prob * 100
        risk_label = risk_score_to_label(risk_percent)

        st.divider()
        st.subheader("Results")

        # Trust level display
        if risk_label == "Low Risk":
            st.success(f"**Trust Level:** {risk_label}")
        elif risk_label == "Medium Risk":
            st.warning(f"**Trust Level:** {risk_label}")
        else:
            st.error(f"**Trust Level:** {risk_label}")

        st.metric("Risk Score", f"{risk_percent:.1f}%")

        st.info(
            "The risk score indicates the model's confidence that the posting exhibits "
            "risk indicators. Higher scores suggest more caution is advised."
        )


if __name__ == "__main__":
    main()

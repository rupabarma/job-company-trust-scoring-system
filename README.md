# Job & Company Trust Scoring System

A proof-of-concept machine learning system that evaluates the trustworthiness of hiring companies or recruiters on job platforms using risk signals. The system outputs a neutral **Trust Level** and **Risk Score**—not a public label—to support Trust & Safety and user due diligence.

---

## Problem Statement

Recruitment scams and fraudulent job postings are a growing concern on professional networks and job platforms. Bad actors may impersonate legitimate companies, request upfront payments, or use personal email domains to appear less trackable. This project explores an ML-based approach to flag risk indicators and surface a trust score for internal or user-facing evaluation.

---

## ML Approach

- **Model:** Random Forest Classifier (scikit-learn)
- **Task:** Binary classification (genuine vs. high-risk profiles)
- **Output:** Predicted class plus probability → converted to a **Risk Score** percentage
- **Language:** Neutral, LinkedIn-safe wording (e.g., "Trust Level", "Risk Indicators", "Proof of Concept")

---

## Features Used

| Feature              | Type    | Description                                      |
|----------------------|---------|--------------------------------------------------|
| `has_website`        | 0/1     | Company has a verifiable website                 |
| `uses_company_email` | 0/1     | Uses company domain email (e.g. @company.com)   |
| `employee_count`     | integer | Approximate number of employees                  |
| `asks_money`         | 0/1     | Requests money or payment upfront               |
| `has_reviews`        | 0/1     | Has verified reviews or ratings                  |

---

## Project Structure

```
Job & Company Trust Scoring System/
├── app.py              # Streamlit UI for inference
├── train_model.py      # Model training script
├── scam_jobs.csv       # Example dataset
├── scam_detector.pkl   # Trained model (created by train_model.py)
├── requirements.txt   # Dependencies
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This creates `scam_detector.pkl` and prints accuracy and classification metrics.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL in your browser, enter the risk indicators, and view the Trust Level and Risk Score.

---

## Future Improvements

- **NLP:** Analyze job description text for red-flag phrases (e.g., "wire transfer", "urgent hiring")
- **Website age:** Integrate domain registration age as a trust signal
- **User reports:** Incorporate platform-level feedback and report volume
- **Feature expansion:** Add domain reputation, LinkedIn company page verification, etc.
- **Calibration:** Improve probability calibration for more reliable risk scores

---

## Disclaimer

This is a **proof-of-concept** and not production-ready. It uses a small synthetic dataset and simplified features. Do not use it as the sole basis for trust or safety decisions. Always conduct your own due diligence when evaluating job opportunities.

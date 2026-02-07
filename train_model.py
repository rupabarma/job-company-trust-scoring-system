"""
Job & Company Trust Scoring System - Model Training Script
Trains a RandomForestClassifier on recruitment risk signals and saves the model.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuration
DATA_PATH = Path(__file__).parent / "scam_jobs.csv"
MODEL_PATH = Path(__file__).parent / "scam_detector.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    """Load the dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    return df


def train_model() -> RandomForestClassifier:
    """Load data, train RandomForest model, save it, and return metrics."""
    print("Loading dataset...")
    df = load_data()

    # Separate features and target
    feature_columns = [
        "has_website",
        "uses_company_email",
        "employee_count",
        "asks_money",
        "has_reviews",
    ]
    X = df[feature_columns]
    y = df["label"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train RandomForest
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "High Risk"]))

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\nModel saved successfully to {MODEL_PATH}")
    print("Training complete.")

    return model


if __name__ == "__main__":
    train_model()

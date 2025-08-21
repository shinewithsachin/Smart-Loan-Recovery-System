import os
import argparse
import json
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from .preprocess import preprocess
from .segment import fit_kmeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train(data_csv: str, model_path: str):
    if not os.path.exists(data_csv):
        logging.error(f"Dataset not found at {data_csv}")
        return

    logging.info(f"Loading dataset from {data_csv}")
    df = pd.read_csv(data_csv)

    df, features = preprocess(df)
    logging.info(f"Preprocessing complete. Using features: {features}")

    if "target_non_recovered" not in df.columns:
        logging.error("Target column 'target_non_recovered' is missing.")
        return

    X = df[features]
    y = df["target_non_recovered"]

    km_feats = ["income_to_emi_ratio", "Days_Past_Due", "Num_Missed_Payments", "loan_to_collateral_ratio"]
    km_feats = [c for c in km_feats if c in X.columns]

    if km_feats:
        logging.info(f"Fitting KMeans on features: {km_feats}")
        fit_kmeans(X[km_feats], k=3)
    else:
        logging.warning("No KMeans features found. Skipping segmentation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logging.info("Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, (y_proba > 0.5).astype(int), output_dict=True)

    logging.info(f"ROC-AUC: {round(auc, 4)}")
    logging.info("Classification Report:")
    logging.info(json.dumps(report, indent=4))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": rf, "features": features}, model_path)
    logging.info(f"Model saved to {model_path}")

    metrics_path = os.path.join(os.path.dirname(model_path), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"roc_auc": auc, "classification_report": report}, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Loan Recovery Model")
    parser.add_argument("--data", type=str, default="data/loan-recovery.csv", help="Path to the dataset CSV file")
    parser.add_argument("--model", type=str, default="models/rf_model.pkl", help="Path to save trained model")
    args = parser.parse_args()

    train(data_csv=args.data, model_path=args.model)

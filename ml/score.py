import pandas as pd
import joblib
from .preprocess import preprocess

MODEL_PATH = "models/rf_model.pkl"
KMEANS_PATH = "models/kmeans.pkl"


def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def score(df: pd.DataFrame):
    # returns df with risk_score, cluster
    df_proc, features = preprocess(df)
    model, feature_list = load_model()
    X = df_proc[feature_list]
    risk_score = model.predict_proba(X)[:, 1]

    # try load kmeans
    try:
        km = joblib.load(KMEANS_PATH)
        km_feats = ["income_to_emi_ratio", "Days_Past_Due", "Num_Missed_Payments", "loan_to_collateral_ratio"]
        clusters = km.predict(df_proc[km_feats])
    except Exception:
        clusters = [-1] * len(df)

    out = df.copy()
    out["risk_score"] = risk_score
    out["cluster"] = clusters
    return out


if __name__ == "__main__":
    df = pd.read_csv("data/loan-recovery.csv")
    out = score(df)
    out.to_csv("data/scored_output.csv", index=False)
    print("Wrote data/scored_output.csv")

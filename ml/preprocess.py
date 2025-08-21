import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CATEGORICAL = [
    "Gender",
    "Employment_Type",
    "Payment_History",
    "Collection_Method",
    "Legal_Action_Taken"
]

def preprocess(df: pd.DataFrame):
    df = df.copy()

    num_cols = [
        "Age", "Monthly_Income", "Num_Dependents", "Loan_Amount", "Loan_Tenure",
        "Interest_Rate", "Collateral_Value", "Outstanding_Loan_Amount", "Monthly_EMI",
        "Num_Missed_Payments", "Days_Past_Due", "Collection_Attempts"
    ]
    
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0  # Create column if missing
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())  # Fill NaNs with median

   
    # Avoid division by zero using replace(0, np.nan)
    df["income_to_emi_ratio"] = df["Monthly_Income"] / (df["Monthly_EMI"].replace(0, np.nan) + 1)
    df["loan_to_collateral_ratio"] = df["Loan_Amount"] / (df["Collateral_Value"].replace(0, np.nan) + 1)
    df["missed_payment_flag"] = (df["Num_Missed_Payments"] > 0).astype(int)
    df["payment_delay_flag"] = (df["Days_Past_Due"] > 30).astype(int)

    
    encoders = {}
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "NA"
        df[col] = df[col].astype(str).fillna("NA")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    
    if "Recovery_Status" in df.columns:
        df["target_non_recovered"] = (~df["Recovery_Status"].str.lower().eq("fully recovered")).astype(int)
    else:
        df["target_non_recovered"] = 0

    
    features = [
        "Age", "Monthly_Income", "Num_Dependents", "Loan_Amount", "Loan_Tenure", "Interest_Rate",
        "Collateral_Value", "Outstanding_Loan_Amount", "Monthly_EMI", "Num_Missed_Payments", "Days_Past_Due",
        "income_to_emi_ratio", "loan_to_collateral_ratio", "missed_payment_flag", "payment_delay_flag"
    ] + CATEGORICAL

    df.fillna(0, inplace=True)
    return df, features

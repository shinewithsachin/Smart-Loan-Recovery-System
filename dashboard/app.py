import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import streamlit as st
import pandas as pd
import plotly.express as px
from ml.score import score
from ml.strategy import apply_strategies


st.set_page_config(page_title="Loan Recovery Dashboard", layout="wide")
st.title("Smart Loan Recovery Dashboard")


uploaded = st.file_uploader("Upload CSV with borrower data", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    scored = score(df)
    scored = apply_strategies(scored)

    st.subheader("Scored Sample")
    st.dataframe(
        scored[["Borrower_ID", "risk_score", "recommended_action", "Days_Past_Due", "Num_Missed_Payments"]].head(50)
    )

    st.subheader("Risk Distribution")
    fig = px.histogram(scored, x="risk_score", nbins=30, marginal="violin", title="Risk Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommended Actions Breakdown")
    counts = scored["recommended_action"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    fig2 = px.bar(counts, x="action", y="count", title="Action Breakdown", text="count")
    st.plotly_chart(fig2, use_container_width=True)

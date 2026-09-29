import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import streamlit as st


def build_clv_table(df):
    """Aggregate per-customer stats and assign CLV tier."""
    clv = df.groupby("User_ID").agg(
        Cust_name=("Cust_name", "first"),
        Gender=("Gender", "first"),
        Age=("Age", "first"),
        State=("State", "first"),
        Occupation=("Occupation", "first"),
        Marital_Status=("Marital_Status", "first"),
        Total_Orders=("Orders", "sum"),
        Total_Amount=("Amount", "sum"),
        Unique_Categories=("Product_Category", "nunique"),
    ).reset_index()

    clv["CLV_Score"] = clv["Total_Orders"] * clv["Total_Amount"]

    # Robust Tier assignment using quantiles with rank-based fallback
    try:
        clv["CLV_Tier"] = pd.qcut(clv["CLV_Score"], q=4, labels=["Low", "Medium", "High", "VIP"], duplicates="drop")
    except ValueError:
        try:
            clv["CLV_Tier"] = pd.cut(clv["CLV_Score"].rank(method="first"), bins=4, labels=["Low", "Medium", "High", "VIP"])
        except Exception:
            clv["CLV_Tier"] = "Medium"

    return clv


def plot_clv_distribution(clv_df):
    """Pie chart of CLV tier distribution."""
    tier_counts = clv_df["CLV_Tier"].value_counts().reset_index()
    tier_counts.columns = ["CLV Tier", "Customers"]
    fig = px.pie(tier_counts, values="Customers", names="CLV Tier",
                 title="Customer Distribution by CLV Tier",
                 color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                 hole=0.4)
    return fig


def plot_clv_tier_stats(clv_df):
    """Bar chart: average spending per CLV tier."""
    stats = clv_df.groupby("CLV_Tier", observed=False).agg(
        Avg_Orders=("Total_Orders", "mean"),
        Avg_Spend=("Total_Amount", "mean"),
        Count=("User_ID", "count"),
    ).reset_index()
    stats["Avg_Orders"] = stats["Avg_Orders"].round(1)
    stats["Avg_Spend"] = stats["Avg_Spend"].round(0).astype(int)
    fig = px.bar(stats, x="CLV_Tier", y="Avg_Spend", color="CLV_Tier",
                 text_auto=True,
                 color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                 title="Average Spending per CLV Tier")
    fig.update_layout(xaxis_title="CLV Tier", yaxis_title="Avg Spend (₹)", showlegend=False)
    return fig, stats


@st.cache_resource
def train_clv_classifier(clv_df):
    """Train a classifier to predict CLV tier from demographics."""
    feature_cols = ["Gender", "Age", "Marital_Status", "State", "Occupation"]
    model_df = clv_df[feature_cols + ["CLV_Tier"]].copy()

    cat_cols = ["Gender", "State", "Occupation"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    # Encode target
    tier_le = LabelEncoder()
    model_df["CLV_Tier"] = tier_le.fit_transform(model_df["CLV_Tier"].astype(str))
    encoders["CLV_Tier"] = tier_le

    X = model_df[feature_cols]
    y = model_df["CLV_Tier"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)

    return clf, encoders, acc

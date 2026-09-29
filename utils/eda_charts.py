import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------- color palette ----------
PALETTE = px.colors.qualitative.Set2


def plot_gender_count(df):
    """Bar chart: number of buyers by gender."""
    counts = df["Gender"].value_counts().reset_index()
    counts.columns = ["Gender", "Count"]
    fig = px.bar(counts, x="Gender", y="Count", color="Gender",
                 text_auto=True, color_discrete_sequence=PALETTE,
                 title="Number of Buyers by Gender")
    fig.update_layout(showlegend=False)
    return fig


def plot_gender_amount(df):
    """Bar chart: total spending by gender."""
    amt = df.groupby("Gender")["Amount"].sum().reset_index()
    fig = px.bar(amt, x="Gender", y="Amount", color="Gender",
                 text_auto=".2s", color_discrete_sequence=PALETTE,
                 title="Total Sales Amount by Gender")
    fig.update_layout(showlegend=False, yaxis_title="Total Amount (₹)")
    return fig


def plot_age_count(df):
    """Bar chart: buyer count by age group."""
    order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
    counts = df["Age Group"].value_counts().reindex(order).reset_index()
    counts.columns = ["Age Group", "Count"]
    fig = px.bar(counts, x="Age Group", y="Count", color="Age Group",
                 text_auto=True, color_discrete_sequence=PALETTE,
                 title="Number of Buyers by Age Group")
    fig.update_layout(showlegend=False)
    return fig


def plot_age_amount(df):
    """Bar chart: total spending by age group."""
    order = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
    amt = df.groupby("Age Group")["Amount"].sum().reindex(order).reset_index()
    fig = px.bar(amt, x="Age Group", y="Amount", color="Age Group",
                 text_auto=".2s", color_discrete_sequence=PALETTE,
                 title="Total Sales Amount by Age Group")
    fig.update_layout(showlegend=False, yaxis_title="Total Amount (₹)")
    return fig


def plot_state_analysis(df, top_n=10):
    """Horizontal bar: top N states by total revenue."""
    state_amt = df.groupby("State")["Amount"].sum().nlargest(top_n).sort_values().reset_index()
    fig = px.bar(state_amt, x="Amount", y="State", orientation="h",
                 text_auto=".2s", color="Amount",
                 color_continuous_scale="Tealgrn",
                 title=f"Top {top_n} States by Revenue")
    fig.update_layout(yaxis_title="", xaxis_title="Total Amount (₹)", coloraxis_showscale=False)
    return fig


def plot_marital_analysis(df):
    """Grouped bar: spending by marital status and gender."""
    grp = df.groupby(["Marital_Status_Label", "Gender"])["Amount"].sum().reset_index()
    fig = px.bar(grp, x="Marital_Status_Label", y="Amount", color="Gender",
                 barmode="group", text_auto=".2s", color_discrete_sequence=PALETTE,
                 title="Spending by Marital Status & Gender")
    fig.update_layout(xaxis_title="Marital Status", yaxis_title="Total Amount (₹)")
    return fig


def plot_occupation_analysis(df, top_n=10):
    """Bar chart: top occupations by revenue."""
    occ = df.groupby("Occupation")["Amount"].sum().nlargest(top_n).reset_index()
    fig = px.bar(occ, x="Occupation", y="Amount", color="Occupation",
                 text_auto=".2s", color_discrete_sequence=PALETTE,
                 title=f"Top {top_n} Occupations by Revenue")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total Amount (₹)")
    return fig


def plot_category_analysis(df):
    """Bar chart: product category performance."""
    cat = df.groupby("Product_Category")["Amount"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(cat, x="Product_Category", y="Amount", color="Product_Category",
                 text_auto=".2s", color_discrete_sequence=PALETTE,
                 title="Sales by Product Category")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total Amount (₹)",
                      xaxis_tickangle=-45)
    return fig


def plot_top_products(df, top_n=10):
    """Bar chart: top N products by order count."""
    prod = df.groupby("Product_ID")["Orders"].sum().nlargest(top_n).sort_values(ascending=True).reset_index()
    fig = px.bar(prod, x="Orders", y="Product_ID", orientation="h",
                 text_auto=True, color="Orders",
                 color_continuous_scale="Sunset",
                 title=f"Top {top_n} Products by Orders")
    fig.update_layout(yaxis_title="Product ID", xaxis_title="Total Orders", coloraxis_showscale=False)
    return fig


def compute_rfm_clusters(rfm_df, n_clusters=4):
    """Run K-Means on Frequency & Monetary. Returns rfm_df with 'Cluster' column + summary."""
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm_df[["Frequency", "Monetary"]])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = km.fit_predict(X)

    # Sort clusters by mean monetary so labels are intuitive
    cluster_order = rfm_df.groupby("Cluster")["Monetary"].mean().sort_values().index.tolist()
    label_map_sorted = {old: new for new, old in enumerate(cluster_order)}
    rfm_df["Cluster"] = rfm_df["Cluster"].map(label_map_sorted)

    meaning = {
        0: "Low-value / Infrequent",
        1: "Mid-value Customers",
        2: "High-frequency & High-spending",
        3: "VIP — Highest Value",
    }
    rfm_df["Segment"] = rfm_df["Cluster"].map(meaning)

    summary = rfm_df.groupby(["Cluster", "Segment"]).agg(
        Customers=("User_ID", "count"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
    ).reset_index()
    summary["Avg_Frequency"] = summary["Avg_Frequency"].round(1)
    summary["Avg_Monetary"] = summary["Avg_Monetary"].round(0).astype(int)

    return rfm_df, summary


def plot_rfm_scatter(rfm_df):
    """Scatter plot of RFM clusters."""
    fig = px.scatter(rfm_df, x="Frequency", y="Monetary", color="Segment",
                     hover_data=["User_ID"],
                     title="Customer Segments (RFM Clustering)",
                     color_discrete_sequence=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
    fig.update_layout(xaxis_title="Frequency (Total Orders)",
                      yaxis_title="Monetary (Total Spend ₹)")
    return fig

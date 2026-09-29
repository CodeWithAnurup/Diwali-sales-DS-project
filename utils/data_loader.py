import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


# Path to the dataset relative to the project root with fallbacks for Cloud deployment
def get_data_path():
    root = os.path.dirname(os.path.dirname(__file__))
    candidates = [
        os.path.join(root, "Dewali sales", "Diwali Sales Data.csv"),
        os.path.join(root, "Diwali sales", "Diwali Sales Data.csv"),
        os.path.join(root, "Dewali sales", "diwali sales data.csv"),
        os.path.join(root, "Diwali Sales Data.csv"),
        os.path.join(root, "data", "Diwali Sales Data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(root, "Dewali sales", "Diwali Sales Data.csv")

DATA_PATH = get_data_path()



def load_and_clean_data():
    """Load the Diwali Sales CSV and perform all cleaning steps."""
    df = pd.read_csv(get_data_path(), encoding="latin1")

    # Drop empty / irrelevant columns
    drop_cols = [c for c in ["Status", "unnamed1"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Drop rows with null Amount
    df.dropna(subset=["Amount"], inplace=True)

    # Convert Amount to int
    df["Amount"] = df["Amount"].astype(int)

    # Clean state names (fix non-breaking spaces)
    df["State"] = df["State"].str.replace("\xa0", " ", regex=False)

    # Map Marital_Status to readable labels
    df["Marital_Status_Label"] = df["Marital_Status"].map({0: "Unmarried", 1: "Married"})

    return df


def get_encoded_features(df):
    """Return a copy of df with categorical columns label-encoded, plus the encoders dict."""
    cat_cols = ["Gender", "Age Group", "State", "Zone", "Occupation", "Product_Category"]
    df_enc = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def build_rfm(df):
    """Build RFM table aggregated per customer (User_ID)."""
    rfm = df.groupby("User_ID").agg(
        Frequency=("Orders", "sum"),
        Monetary=("Amount", "sum"),
        Num_Categories=("Product_Category", "nunique"),
    ).reset_index()
    return rfm

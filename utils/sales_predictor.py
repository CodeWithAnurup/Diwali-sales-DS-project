import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st


FEATURE_COLS = ["Gender", "Age", "Marital_Status", "State", "Zone",
                "Occupation", "Product_Category", "Orders"]
TARGET = "Amount"


@st.cache_resource
def train_sales_model(df):
    """Train a Random Forest Regressor to predict Amount."""
    model_df = df[FEATURE_COLS + [TARGET]].copy()

    # Encode categoricals
    cat_cols = ["Gender", "State", "Zone", "Occupation", "Product_Category"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "R² Score": round(r2_score(y_test, y_pred), 4),
        "MAE (₹)": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE (₹)": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
    }

    # Feature importances
    importance = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, encoders, metrics, importance


def predict_single(model, encoders, input_dict):
    """Predict Amount for a single customer input."""
    row = {}
    for col in FEATURE_COLS:
        val = input_dict[col]
        if col in encoders:
            val = encoders[col].transform([str(val)])[0]
        row[col] = val
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    return round(pred, 2)

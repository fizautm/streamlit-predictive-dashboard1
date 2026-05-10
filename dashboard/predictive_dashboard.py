import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Revenue Prediction Dashboard")

# Load trained model using absolute path
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "revenue_model.pkl"

model = joblib.load(model_path)

st.sidebar.header("Input Parameters")

units = st.sidebar.slider("Units Sold", 10, 100)

region = st.sidebar.selectbox(
    "Region",
    ["North", "South", "East", "West"]
)

product = st.sidebar.selectbox(
    "Product",
    ["Widget", "Gadget", "Tool", "Device"]
)

# Prepare input dataframe
input_df = pd.DataFrame({
    "units_sold": [units],
    "region": [region],
    "product": [product]
})

# Predict
predicted_revenue = model.predict(input_df)[0]

st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")


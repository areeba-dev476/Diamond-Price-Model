import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("diamond_price_model.joblib")


model = load_model()

st.title("💎 Diamond Price Predictor")
st.caption("AIC354 Machine Learning Fundamentals – Assignment 3. "
           "Enter the diamond's attributes to get an estimated price in USD.")

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat", min_value=0.2, max_value=5.5, value=0.9, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Colour (D = best, J = worst)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=3)
with col2:
    depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table = st.number_input("Table %", min_value=40.0, max_value=95.0, value=57.0, step=0.1)
    x = st.number_input("Length x (mm)", min_value=0.1, max_value=11.0, value=6.1, step=0.01)
    y = st.number_input("Width y (mm)", min_value=0.1, max_value=60.0, value=6.15, step=0.01)
    z = st.number_input("Depth z (mm)", min_value=0.1, max_value=32.0, value=3.78, step=0.01)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{"carat": carat, "depth": depth, "table": table,
                         "x": x, "y": y, "z": z,
                         "cut": cut, "color": color, "clarity": clarity}])
    price = float(model.predict(row)[0])
    st.success(f"Estimated price: **${price:,.0f}**")

st.divider()
st.caption("Model trained on the full diamonds dataset (53,940 rows) and selected using 10-fold cross-validation.")

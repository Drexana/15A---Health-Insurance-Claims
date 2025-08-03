import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

#Cache models and load them once

@st.cache_resource
def load_models():
  kmeans = joblib.load("models/kmeans_model.joblib")
  scaler = joblib.load("models/scaler.joblib")
  rf_model = joblib.load("models/rf_model.joblib")
  iso_models = joblib.load("models/iso_models.joblib")
  return kmeans, scaler, rf_model, iso_models

kmeans, scaler, rf_model, iso_models = load_models()

cluster_labels = {0: "Preferred", 1: "High-Cost", 2: "Standard"}

st.title("Insurance Risk Analysis Dashboard")
st.markdown("Single Claimant Input")

age = st.number_input("Age", min_value = 18, max_value = 100, value = 35)
smoker = st.selectbox("Smoker", ["Yes", "No"])
bmi = st.number_input("BMI", min_value = 10.0, max_value = 60.0, value = 30.0)

smoker_value = 1.0 if smoker == "Yes" else 0.0

input_df = pd.DataFrame([[age, smoker_value, charges, bmi]], columns = ["age", "smoker", "charges", "bmi"])
scaled_input = scaler.transform(input_df)
cluster_id = kmeans.predict(scaled_input)[0]
cluster_name = cluster_labels.get(cluster_id, "Unknown")
rf_prediction = rf_model.predict(input_df)[0]
iso_model = iso_models.get(cluster_id)
if iso_model:
    anomaly_flag = iso_model.predict(input_df)[0]
    anomaly_score = iso_model.decision_function(input_df)[0]
else:
    anomaly_flag = None

st.subheader("Results for single claimant")

st.write(f"**Cluster Assignment:** {cluster_id} ({cluster_name})")
st.write(f"**Random Forest risk prediction:** {rf_prediction}")








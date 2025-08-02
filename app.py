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



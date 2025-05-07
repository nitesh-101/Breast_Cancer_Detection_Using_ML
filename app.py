import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from bcd import load_data, explore_data, explore_histograms, predict
from model_training import train_and_evaluate_models

# --- Custom Theme Styling ---
st.markdown("""
    <style>
        /* Main page background */
        .stApp { 
            background-color: #ACD9D6 !important;
            color: #1f2937 !important;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #D2DBE5 !important;
        }

        /* Sidebar text color */
        section[data-testid="stSidebar"] * {
            color: #1f2937 !important;
        }

        /* Top bar (header) background */
        header {
            background-color: #008080 !important;
            color: white !important;
        }

        /* Top bar menu text color */
        header .css-1v0mbdj {
            color: white !important;
        }

        /* Radio bullet color */
        div[data-baseweb="radio"] > div > label > div:first-child {
            border: 2px solid #0288d1 !important;
        }
        div[data-baseweb="radio"] > div > label > div:first-child svg {
            color: #0288d1 !important;
        }

        /* Sliders */
        .stSlider > div[data-baseweb="slider"] > div {
            color: #1976d2 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #1976d2 !important;
            color: white !important;
        }

        /* Top-right menu dot */
        header .st-emotion-cache-1avcm0n, .st-emotion-cache-6qob1r {
            color: #0288d1 !important;
        } 
    </style>
""", unsafe_allow_html=True)

# Load Dataset
features, labels, df, feature_names, label_names = load_data()

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("**Choose Section ⬇️**", ["🏠 Home", "📊 Data Exploration", "🧠 Model Training", "🩺 Prediction"])

# Home
if page == "🏠 Home":
    st.title("Breast Cancer Detection App 🎗️")
    
    st.markdown("""
    **Breast Cancer** is one of the most common types of cancer worldwide. Early detection can save lives, making it crucial to identify whether a tumor is benign or malignant as soon as possible.
    """)
    
    # Display the image
    st.image("image.png", use_container_width=True)

    st.markdown("""
    This app leverages machine learning models to predict whether a tumor is **Benign** or **Malignant** based on diagnostic features.

    **Early detection saves lives**—start exploring today.\n
    Use the menu on the left to **explore data**, **train models**, or **make predictions**.
    """)

# Data Exploration
elif page == "📊 Data Exploration":
    explore_data(df, label_names)
    explore_histograms(df)

# Model Training
elif page == "🧠 Model Training":
    train_and_evaluate_models(features, labels, feature_names, label_names)

# Prediction
elif page == "🩺 Prediction":
    model = joblib.load("breast_cancer_rf_model.pkl")
    st.title("Make a Prediction")
    st.subheader("Input Diagnostic Features")

    user_input = []
    min_vals = df[feature_names].min()
    max_vals = df[feature_names].max()
    mean_vals = df[feature_names].mean()

    # Create 3 columns for 10 features each
    col1, col2, col3 = st.columns(3)

    # Split features into 3 equal parts
    chunk_size = len(feature_names) // 3
    features_col1 = feature_names[:chunk_size]
    features_col2 = feature_names[chunk_size:2*chunk_size]
    features_col3 = feature_names[2*chunk_size:]

    with col1:
        for feature in features_col1:
            val = st.slider(f"{feature}", float(min_vals[feature]), float(max_vals[feature]), float(mean_vals[feature]), step=0.1)
            user_input.append(val)

    with col2:
        for feature in features_col2:
            val = st.slider(f"{feature}", float(min_vals[feature]), float(max_vals[feature]), float(mean_vals[feature]), step=0.1)
            user_input.append(val)

    with col3:
        for feature in features_col3:
            val = st.slider(f"{feature}", float(min_vals[feature]), float(max_vals[feature]), float(mean_vals[feature]), step=0.1)
            user_input.append(val)

    if st.button("Predict"):
        predict(np.array(user_input).reshape(1, -1), model, label_names)

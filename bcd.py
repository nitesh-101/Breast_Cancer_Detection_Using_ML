import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return data['data'], data['target'], df, data['feature_names'], data['target_names']

def explore_data(df, label_names):
    st.title("Data Exploration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tumor Type Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='target', data=df, ax=ax1)
        ax1.set_xticklabels(label_names)
        st.pyplot(fig1)

    with col2:
        st.subheader("Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

def explore_histograms(df):
    st.subheader("Histograms for Selected Features")
    selected_features = ['mean radius', 'mean texture', 'mean concavity', 'worst radius', 'worst texture', 'worst concavity']
    fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle('Distributions of Selected Features by Tumor Type', fontsize=16)

    for i, feature in enumerate(selected_features):
        row, col = divmod(i, 3)
        sns.histplot(data=df, x=feature, hue='target', kde=True, ax=axes[row, col])
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel('')
        axes[row, col].set_ylabel('')

    fig3.legend(labels=['Malignant', 'Benign'], loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    st.pyplot(fig3)

def predict(input_array, model, label_names):
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    label = label_names[prediction].capitalize()

    if prediction == 0:  # Malignant
        st.markdown(f"<div style='background-color: #ff6666; color: black; padding: 10px; border-radius: 5px;'>Prediction: {label}</div>", unsafe_allow_html=True)
    else:  # Benign
        st.markdown(f"<div style='background-color: #66ff66; color: black; padding: 10px; border-radius: 5px;'>Prediction: {label}</div>", unsafe_allow_html=True)

    st.write(f"**Probability of Benign:** {probabilities[1]:.4f}")
    st.write(f"**Probability of Malignant:** {probabilities[0]:.4f}")

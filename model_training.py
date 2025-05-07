import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import streamlit as st

def train_and_evaluate_models(features, labels, feature_names, label_names):
    st.title("Model Training & Comparison")
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    models = {
        'Gaussian Naive Bayes': (GaussianNB(), {}),
        'Logistic Regression': (LogisticRegression(solver='liblinear', max_iter=1000), {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}),
        'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 150]})
    }

    comparison_metrics = {}

    for name, (model, grid_params) in models.items():
        if grid_params:
            grid_search = GridSearchCV(model, grid_params, cv=5, scoring='f1_weighted')
            grid_search.fit(train_features, train_labels)
            best_model = grid_search.best_estimator_
        else:
            model.fit(train_features, train_labels)
            best_model = model

        preds = best_model.predict(test_features)
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, average='macro')
        precision = precision_score(test_labels, preds)
        recall = recall_score(test_labels, preds)
        comparison_metrics[name] = [acc, f1, precision, recall]

        if name == "Random Forest":
            joblib.dump(best_model, 'breast_cancer_rf_model.pkl')

    st.subheader("Model Performance Comparison")
    metric_df = pd.DataFrame(comparison_metrics, index=['Accuracy', 'F1-Score', 'Precision', 'Recall']).T
    st.dataframe(metric_df.style.format("{:.4f}"))
    st.bar_chart(metric_df)

    # Side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Important Features (Logistic Regression)")
        log_model = LogisticRegression(solver='liblinear')
        log_model.fit(train_features, train_labels)
        importance = np.abs(log_model.coef_[0])
        feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)[:10]
        fig4, ax4 = plt.subplots()
        sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax4)
        st.pyplot(fig4)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(test_labels, preds)
        fig5, ax5 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=ax5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('True')
        st.pyplot(fig5)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(test_labels, preds, target_names=label_names)
    st.text(report)

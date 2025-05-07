# 🎗️ Breast Cancer Detection Using Machine Learning
A Streamlit web application that uses **Machine Learning models** to predict whether a tumor is **Benign** or **Malignant** based on diagnostic features from the breast cancer dataset.
![image alt](https://github.com/nitesh-101/Breast_Cancer_Detection_Using_ML/blob/main/homepage.png?raw=true)

## Introduction 

Breast cancer is a significant health concern worldwide. Leveraging machine learning can provide a valuable tool for assisting medical professionals in early diagnosis, potentially leading to timely interventions and improved survival rates. This project aims to:
* Explore the effectiveness of various machine learning algorithms in breast cancer detection.
* Build a robust and reliable predictive model.
* Provide a clear and understandable implementation for educational and research purposes.

## Dataset

* Dataset: Wisconsin Breast Cancer (Diagnostic) Dataset.
* Source: scikit-learn library (sklearn.datasets).
* Data: Features from digitized images of breast mass Fine Needle Aspirates (FNA).
* Features: 30 features describing cell nuclei (e.g., radius, texture), with mean, standard error, and "worst" values.
* Instances: 569.
* Target Variable: Binary (malignant or benign diagnosis).

## Methodology
The project follows a standard machine learning workflow:

1. **Data Exploration and Preprocessing:**
The dataset was loaded and inspected for structure and quality. Exploratory Data Analysis (EDA) was used to visualize feature distributions and correlations. Missing values were checked (none found), and since the data was already numeric and scaled, minimal preprocessing was needed. The data was then split into training and testing sets.

2. **Model Selection and Training:**
Multiple classification algorithms were explored including Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, and Random Forests. Models were trained on the training set, and hyperparameter tuning was performed using GridSearchCV to improve performance.

3. **Model Evaluation:**
Trained models were evaluated on test data using various metrics such as Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Area Under the ROC Curve (AUC) to determine the best-performing approach.

## Result
Model Performance:
The models were evaluated using accuracy, precision, recall, and F1-score.Among the tested algorithms, Random Forest achieved the highest performance.
* Model Metrics:
     * Accuracy: 97.37%     
     * Precision: 97.06%     
     * Recall: 98.04%      
     * F1-Score: 97.55%

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/nitesh-101/Breast_Cancer_Detection_Using_ML.git
   cd Breast_Cancer_Detection_App

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the app:
   ```bash
   streamlit run app.py

📄 License
This project is licensed under the MIT License - see the [MIT License](LICENSE) file for details.
                                                                                                                   


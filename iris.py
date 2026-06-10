import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error

# 1. Suppress all unnecessary warning outputs in the UI
import warnings
warnings.filterwarnings('ignore')

# App Configuration & Title
st.set_page_config(page_title="CodeAlpha ML Intern App", layout="wide")
st.title("🚀 Data Science Internship Project Dashboard")
st.markdown("Developed for the **CodeAlpha Data Science Internship** core requirements.")

# Sidebar Navigation
st.sidebar.header("📁 Navigation Menu")
project_choice = st.sidebar.radio("Select a Project to View:", ["🌸 Task 1: Iris Classification", "🚗 Task 3: Car Price Prediction"])

# ==========================================
# TASK 1: IRIS FLOWER CLASSIFICATION
# ==========================================
if project_choice == "🌸 Task 1: Iris Classification":
    st.header("🌸 Iris Flower Classification (Logistic Regression)")
    st.write("This tab fits a multi-class Logistic Regression pipeline to classify species based on structural measurements.")

    # Load Dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(X.head(), use_container_width=True)

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Clean multi_class parameter definition to avoid FutureWarnings
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Interactive User Inputs for Predictions
    st.subheader("🔮 Live Prediction Sandbox")
    st.write("Adjust the sliders below to see the trained Logistic Regression model classify the species in real time:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        s_len = st.slider("Sepal Length (cm)", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), float(X.iloc[:,0].mean()))
    with col2:
        s_wid = st.slider("Sepal Width (cm)", float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), float(X.iloc[:,1].mean()))
    with col3:
        p_len = st.slider("Petal Length (cm)", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), float(X.iloc[:,2].mean()))
    with col4:
        p_wid = st.slider("Petal Width (cm)", float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), float(X.iloc[:,3].mean()))

    # Run custom user prediction
    user_input = np.array([[s_len, s_wid, p_len, p_wid]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    predicted_species = iris.target_names[prediction]

    st.success(f"🎯 **Predicted Species:** {predicted_species.upper()}")

    # Model Performance Evaluation Metrics
    st.subheader("📊 Model Performance Summary")
    metric_col1, metric_col2 = st.columns([1, 2])
    
    with metric_col1:
        st.metric(label="Model Accuracy Score", value=f"{acc * 100:.2f}%")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, target_names=iris.target_names))
        
    with metric_col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples',
                    xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
        plt.title('Confusion Matrix')
        st.pyplot(fig)
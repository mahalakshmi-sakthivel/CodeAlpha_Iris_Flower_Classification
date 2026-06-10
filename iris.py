import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Suppress all unnecessary deprecation or version warnings in the terminal/UI
import warnings
warnings.filterwarnings('ignore')

# 2. Page Configuration
st.set_page_config(page_title="Iris Classification Dashboard", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classification Dashboard")
st.markdown("Developed for the **CodeAlpha Data Science Internship**[cite: 1, 3]. Uses **Logistic Regression** to classify species[cite: 28].")

# 3. Load the Dataset directly from a reliable online source (No local CSV required!)
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    colnames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris_df = pd.read_csv(url, header=None, names=colnames)
    return iris_df

try:
    df = load_data()
    
    # Split features and target
    X = df.drop(columns=['species'])
    y = df['species']
    
    # Layout configuration: Split screen into Left and Right columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Dataset Preview (First 5 Rows)")
        st.dataframe(X.head(), use_container_width=True)

    # 4. Train the Machine Learning Pipeline 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a clean Logistic Regression model without deprecated parameters [cite: 28]
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) [cite: 30]

    # 5. Interactive User Input Sliders for Live Predictions [cite: 27]
    with col_right:
        st.subheader("🔮 Live Prediction Interactive Sandbox")
        st.write("Adjust these sliders to send live measurements into the trained model[cite: 27, 28]:")
        
        s_len = st.slider("Sepal Length (cm)", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), float(X.iloc[:,0].mean()))
        s_wid = st.slider("Sepal Width (cm)", float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), float(X.iloc[:,1].mean()))
        p_len = st.slider("Petal Length (cm)", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), float(X.iloc[:,2].mean()))
        p_wid = st.slider("Petal Width (cm)", float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), float(X.iloc[:,3].mean()))

        # Format live data input matrix for prediction
        user_data = np.array([[s_len, s_wid, p_len, p_wid]])
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]

        # Display the result inside a beautiful highlighted block [cite: 28]
        st.success(f"🎯 **Model Classified Species As:** {prediction.replace('Iris-', '').upper()}")

    st.markdown("---")

    # 6. Model Performance Evaluation Display Metrics [cite: 30]
    st.subheader("📊 Model Performance Evaluation")
    metric_col1, metric_col2 = st.columns([1, 1])

    with metric_col1:
        st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%") [cite: 30]
        st.text("Detailed Classification Report:")
        st.code(classification_report(y_test, y_pred))

    with metric_col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        # Get unique target names from model classes
        labels = [label.replace('Iris-', '') for label in model.classes_]
        
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        plt.title('Classification Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(fig)

except Exception as e:
    st.error(f"🌐 Connection or Loading Error: Make sure your computer is connected to the internet to stream the dataset. Detail: {e}")

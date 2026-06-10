import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Page Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# 2. Title and Description
st.title("🌸 Iris Flower Classification App")
st.markdown("""
This app predicts the **Iris flower species** based on sepal and petal measurements using a Machine Learning model.
""")

# 3. Load and Prepare Dataset
@st.cache_data # Caches the data loading process for faster performance
def load_and_train():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    return model, iris.target_names, X, acc

model, target_names, X_data, model_accuracy = load_and_train()

# Display Model Accuracy in the app
st.sidebar.success(f"Model Accuracy: {model_accuracy * 100:.2f}%")

# 4. User Input Parameters (Sidebar Sliders)
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X_data.iloc[:, 0].min()), float(X_data.iloc[:, 0].max()), float(X_data.iloc[:, 0].mean()))
    sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X_data.iloc[:, 1].min()), float(X_data.iloc[:, 1].max()), float(X_data.iloc[:, 1].mean()))
    petal_length = st.sidebar.slider("Petal Length (cm)", float(X_data.iloc[:, 2].min()), float(X_data.iloc[:, 2].max()), float(X_data.iloc[:, 2].mean()))
    petal_width = st.sidebar.slider("Petal Width (cm)", float(X_data.iloc[:, 3].min()), float(X_data.iloc[:, 3].max()), float(X_data.iloc[:, 3].mean()))
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_user_input = user_input_features()

# 5. Display User Input
st.subheader("User Input Parameters")
st.write(df_user_input)

# 6. Make Prediction
prediction = model.predict(df_user_input)
prediction_proba = model.predict_proba(df_user_input)

# 7. Display Results
st.subheader("Prediction")
predicted_species = target_names[prediction[0]].capitalize()
st.metric(label="Predicted Species", value=predicted_species)

st.subheader("Prediction Probability")
prob_df = pd.DataFrame(prediction_proba, columns=[name.capitalize() for name in target_names])
st.dataframe(prob_df)
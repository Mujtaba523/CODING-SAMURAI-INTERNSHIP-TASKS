# Importing Libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import json
from streamlit_lottie import st_lottie

# Utility Functions
def load_lottie(filepath):
    """
    Load and return the contents of a Lottie file.
    
    Args:
        filepath (str): Path to the Lottie file.
    
    Returns:
        dict: JSON contents of the Lottie file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)
    
def load_models():
    """
    Load and return the trained models used for weather classification.
    
    Returns:
        dict: Dictionary containing the loaded models.
    """
    models = {
        "Decision Tree": joblib.load("Models/dt.pkl"),
        "Logistic Regression": joblib.load("Models/lr.pkl"),
        "Extra Trees": joblib.load("Models/et.pkl"),
        "Naive Bayes": joblib.load("Models/nb.pkl"),
    }
    return models

# Initial configuration
st.set_page_config(
    page_title="🍀BlossomBot: A Flower-Based Classifier",
    initial_sidebar_state="expanded",
    layout='wide',
) 
# Loading models, images and scaler
models = load_models()
scaler = joblib.load("Models/scaler.pkl")
class_images = {
    "Iris-setosa": "Vectors/setosa.jpg",
    "Iris-versicolor": "Vectors/versicolor.jpg",
    "Iris-virginica": "Vectors/Virginica.jpg",
}
# Class Mapping
class_name = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
# Streamlit Application
st.title("🍀BlossomBot: A Flower-Based Classifier")
st.markdown("#### It is an advanced machine learning application designed to accurately identify and classify various types of flowers. Leveraging state-of-the-art ML algorithms and trained on the renowned Iris dataset, BlossomBot is your go-to tool for precise flower recognition.")
left, right = st.columns(2)
with left:
    # Input Widgets
    sepal_width = st.slider("Sepal Width:", 0.0, 10.0, 5.0, 0.1)
    sepal_length = st.slider("Sepal Length:", 0.0, 10.0, 3.5, 0.1)
    petal_width = st.slider("Petal Width:", 0.0, 10.0, 1.4, 0.1)
    petal_length = st.slider("Petal Length:", 0.0, 10.0, 0.2, 0.1)
    input_data = pd.DataFrame({"sepal_width":[sepal_width], "sepal_length":[sepal_length], "petal_width":[petal_width], "petal_length":[petal_length]})
    scaler = joblib.load("Models/scaler.pkl")
    scaled_input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
    model_name = st.selectbox("Select Model:", ["Decision Tree", "Logistic Regression", "Extra Trees", "Naive Bayes"])
    predict_button = st.button("Predict")
    # Output
    if predict_button:
        semi_left, semi_right = st.columns(2)
        with semi_left:
            predictions = models[model_name].predict(scaled_input_data)[0]
            probabilities = models[model_name].predict_proba(scaled_input_data)
            max_prob_indices = np.argmax(probabilities, axis=1)
            for i, max_prob_index in enumerate(max_prob_indices):
                max_prob = probabilities[i, max_prob_index]
                # Print the highest probability and its corresponding class label
                st.write(f"##### The classified weather summary is: {class_name[predictions]}")
                st.write(f"##### The classified probability is: {max_prob*100:.2f}%")
            with semi_right:
                image_path = Image.open(class_images[class_name[predictions]]).resize((300,200))
                st.image(image_path) 
with right:
    animation1 = load_lottie("Vectors/2.json")
    st_lottie(animation1, loop=True, width=800, height=600)
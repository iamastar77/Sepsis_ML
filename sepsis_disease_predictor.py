import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# Load the models
RF = joblib.load('rf.pkl')
LR = joblib.load('log_reg.pkl')
LightGBM = joblib.load('lgb.pkl')

# Model dictionary
models = {
    'Random Forest': RF,
    'Logistic Regression': LR,
    'LightGBM': LightGBM
}

# Title
st.title("Sepsis Prediction App")

# Description
st.write("""
This app predicts the likelihood of sepsis based on input features.
Select one or more models, input feature values, and get predictions and probability estimates.
""")

# Sidebar for model selection with multi-select option
selected_models = st.sidebar.multiselect("Select models to use for prediction", list(models.keys()),
                                         default=list(models.keys()))

# Input fields for the features
st.sidebar.header("Enter the following feature values:")
age = st.sidebar.number_input("Age", min_value=18.0, max_value=90.0, value=50.0)
mapp = st.sidebar.number_input("Mean Arterial Pressure", min_value=0.0, max_value=200.0, value=80.0)
temp = st.sidebar.number_input("Body Temperature", min_value=30.0, max_value=42.0, value=36.5)
rdw = st.sidebar.number_input("Red Blood Cell Distribution Width", min_value=0.0, max_value=25.0, value=13.0)
bun = st.sidebar.number_input("Blood Urea Nitrogen", min_value=0.0, max_value=200.0, value=13.5)
wbc = st.sidebar.number_input("White Blood Cell Count", min_value=0.0, max_value=100.0, value=7.5)
pltt = st.sidebar.number_input("Platelets", min_value=0.0, max_value=700.0, value=275.0)


# Convert inputs to DataFrame for model prediction
input_data = pd.DataFrame({
    'Age': [age],
    'MAP': [mapp],
    'Temp': [temp],
    'RDW': [rdw],
    'BUN': [bun],
    'WBC': [wbc],
    'PLT': [pltt],
})

# Add a predict button
if st.sidebar.button("Predict"):
    # Display predictions and probabilities for selected models
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Display the prediction and probabilities for each selected model
        st.write(f"## Model: {model_name}")
        st.write(f"**Prediction**: {'Sepsis' if prediction == 1 else 'No Sepsis'}")
        st.write("**Prediction Probabilities**")
        st.write(f"Probability of No Sepsis: {probabilities[0]:.4f}")
        st.write(f"Probability of Sepsis: {probabilities[1]:.4f}")

# Display PNG images
st.subheader("1. Radar chart of the performance of each model in the training set")
image1 = Image.open("Training Set Metrics.png")
st.image(image1, caption="Radar chart of the performance of each model in the training set", use_column_width=True)

st.subheader("2. Radar chart of the performance of each model in the test set")
image2 = Image.open("Test Set Metrics.png")
st.image(image2, caption="Radar chart of the performance of each model in the test set", use_column_width=True)
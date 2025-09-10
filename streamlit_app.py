import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"D:\pneumonia\saved_cnn_model.keras")

st.title("Lung X-ray Pneumonia Classifier")
st.write("Upload a lung X-ray image and enter patient details to get a prediction and precautions.")

uploaded_file = st.file_uploader("Upload Lung X-ray Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
fever = st.selectbox("Does the patient have fever?", ["No", "Yes"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape (1, 128, 128, 1)
    patient_data = np.array([[age, 1 if fever == "Yes" else 0]])
    prediction = model.predict([img_array, patient_data])[0][0]
    condition = "bacteria" if prediction > 0.5 else "virus"
    st.subheader(f"Predicted Condition: {condition.capitalize()}")
    st.write("Precautions:")
    precautions = []
    if condition == "virus":
        precautions.append("Rest and stay hydrated to support recovery from viral pneumonia.")
        precautions.append("Consult a doctor if symptoms worsen (e.g., difficulty breathing).")
        if fever == "Yes": precautions.append("Use fever-reducing medication as advised.")
        if age > 60: precautions.append("Seek immediate medical attention due to age risk.")
    elif condition == "bacteria":
        precautions.append("Seek antibiotic treatment from a healthcare provider.")
        precautions.append("Avoid spreading infection by isolating if possible.")
        if fever == "Yes": precautions.append("Monitor fever closely and use medication under guidance.")
        if age > 60: precautions.append("Urgent medical evaluation recommended.")
    for p in precautions:
        st.write(f"- {p}")
else:
    st.info("Please upload a lung X-ray image to get started.")

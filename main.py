import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('acne_detection_model.keras')  # Ensure this file exists in your working directory

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (100, 100))  # Resize to model input size
    img = img / 255.0  # Normalize the image
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Streamlit UI
st.title("Acne Detection App")
st.write("Upload an image to check if it contains acne.")

# Handle uploaded images
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    prediction_class = (prediction > 0.5).astype(int)

    # Show prediction result
    if prediction_class[0][0] == 1:
        st.write("Prediction: The image contains acne.")
    else:
        st.write("Prediction: The image does not contain acne.")

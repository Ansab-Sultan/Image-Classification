import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load the trained model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Load the class names
class_names = ["Benign", "Malignant"] 

# Function for preprocessing image
def preprocess_image(image):
    image = tf.image.resize(image, (144, 144))
    image = tf.cast(image, tf.float32)
    image /= 255.0  # Rescale pixel values to [0, 1]
    return image

# Function for making predictions
def predict(model, image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Title of the web app
st.title('Image Classification App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = preprocess_image(np.array(image))

    # Load the trained model
    model = load_model('trained_model.h5')  # Update with your model path

    # Make predictions
    predicted_class, confidence = predict(model, image)
    
    # Display predictions
    st.write(f"Predicted class: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")

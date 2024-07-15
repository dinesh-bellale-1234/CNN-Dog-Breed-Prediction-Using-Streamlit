import numpy as np
import cv2 # type: ignore
from PIL import Image
import streamlit as st
import pickle
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore

# Load VGG16 model pre-trained on ImageNet, without the top fully connected layers
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Display the file uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Resize the image to 224x224
    resized_image = cv2.resize(image, (224, 224))
    
    # Convert the resized image to RGB format for displaying with Streamlit
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Display the resized image
    st.image(resized_image_rgb, caption='Resized Image (224x224)', use_column_width=True)
    
    # Prepare the image for feature extraction
    image_array = np.expand_dims(resized_image_rgb, axis=0)
    image_array = preprocess_input(image_array)
    
    # Extract features using VGG16
    features = vgg16.predict(image_array)
    
    # Display the Predict button
    if st.button('Predict'):
        # Load the final model for making predictions
        with open(r"C:\Users\deepa\python & ML documents\Machine learning & Deep Learning Notes\Deep Learning Assignments\best_modelsvc.pkl", 'rb') as file:
            final_model = pickle.load(file)
        
        # Make predictions using the extracted features
        features_flattened = features.reshape(features.shape[0], -1)  # Flatten the features
        predictions = final_model.predict(features_flattened)
        
        # Display the prediction results
        st.write("Predictions:", predictions)

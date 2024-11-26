# import streamlit as st
# from PIL import Image
# import os
# import base64
# # Recommended: Use a relative path or place image in a specific directory
# background_image_path = "background.jpg"

# # Add error handling and fallback
# if os.path.exists(background_image_path):
#     try:
#         # Read and encode the image
#         with open(background_image_path, "rb") as image_file:
#             encoded_image = base64.b64encode(image_file.read()).decode()
        
#         # Apply background with blur effect
#         st.markdown(
#             f"""
#             <style>
#             .stApp {{
#                 background-image: url('data:image/jpeg;base64,{encoded_image}');
#                 background-size: cover;
#                 background-position: center;
#                 background-repeat: no-repeat;
#                 background-attachment: fixed;
#                 backdrop-filter: blur(5px);
#                 -webkit-backdrop-filter: blur(5px);
#                 background-color: rgba(255, 255, 255, 0.2);
#             }}
            
#             /* Ensure content remains sharp */
#             .stApp > div {{
#                 backdrop-filter: none;
#                 -webkit-backdrop-filter: none;
#             }}
#             </style>
#             """, 
#             unsafe_allow_html=True
#         )
#     except Exception as e:
#         st.error(f"Error processing background image: {e}")
# else:
#     st.error(f"Background image not found: {background_image_path}")

# # Main Page Content
# st.title("Welcome to the Chest X-Ray Classification Website")
# st.write("""
# This application allows users to upload chest X-ray images and classify them into disease categories:
# - Viral Pneumonia
# - Bacterial Pneumonia
# - Covid
# - Tuberculosis

# Upload your X-ray image to get started.
# """)

# # Sidebar for Doctor's Details
# st.sidebar.title("Doctor's Details")
# doctor_image_path = r"C:\Users\vansh\Desktop\download.jpeg"
# try:
#     st.sidebar.image(doctor_image_path, caption="Dr. John Doe", use_column_width=True)
# except FileNotFoundError:
#     st.sidebar.error("Doctor's image file not found. Please check the path.")
# st.sidebar.write("""
# **Name:** Dr. John Doe  
# **Specialization:** Pulmonologist  
# **Experience:** 15+ years  
# **Contact:** +1 234 567 890  
# **Email:** johndoe@example.com  
# **Clinic:** HealthCare Hospital, New York  
# """)

# # Optional action on the sidebar
# if st.sidebar.button("Contact Doctor"):
#     st.sidebar.write("Visit our contact page for more details!")

# # File uploader for X-ray image
# uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # Placeholder for classification result
#     st.write("**Prediction:** [Model's predicted class will be displayed here]")

# # Footer Section
# st.markdown("---")  # Separator line
# st.markdown(
#     """
#     <style>
#     .footer {
#         text-align: center;
#         font-size: 14px;
#         color: gray;
#         margin-top: 50px;
#     }
#     </style>
#     <div class="footer">
#         <p>Developed by:</p>
#         <p><b>Vanshika Mittal, Nikhil Chadha, Vanshika Narang and Puru Sachdeva</b></p>
#         <p>&copy; 2024 DeepLearning Team</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

import streamlit as st
from PIL import Image
import os
import base64
import numpy as np
from tensorflow.keras.models import load_model  # Replace with PyTorch if needed
from tensorflow.keras.preprocessing.image import img_to_array

# Path to your saved model
model_path = "lungs_classification_model_densenet.h5"

# Load the trained model
try:
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Background setup
background_image_path = "background.jpg"
if os.path.exists(background_image_path):
    try:
        with open(background_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.75), rgba(255, 255, 255, 0.75)),url('data:image/jpeg;base64,{encoded_image}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                backdrop-filter: blur(5px);
                -webkit-backdrop-filter: blur(5px);
                background-color: rgba(255, 255, 255, 0.2);
            }}
            .stApp > div {{
                backdrop-filter: none;
                -webkit-backdrop-filter: none;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error processing background image: {e}")
else:
    st.error(f"Background image not found: {background_image_path}")

# Main Page Content
st.title("Welcome to the Chest X-Ray Classification Website")
st.write(
    """
This application allows users to upload chest X-ray images and classify them into disease categories:
- COVID19
- NORMAL
- VIRAL PNEUMONIA
- BACTERIAL PNEUMONIA
- TUBERCULOSIS

Upload your X-ray image to get started.
"""
)

# Sidebar for Doctor's Details
st.sidebar.title("Doctor's Details")
doctor_image_path = r"WhatsApp Image 2024-11-26 at 07.45.15_5588cb42.jpg"
try:
    st.sidebar.image(doctor_image_path, caption="Dr. John Doe", use_column_width=True)
except FileNotFoundError:
    st.sidebar.error("Doctor's image file not found. Please check the path.")
st.sidebar.write(
    """
*Name:* Dr. John Doe  
*Specialization:* Pulmonologist  
*Experience:* 15+ years  
*Contact:* +1 234 567 890  
*Email:* johndoe@example.com  
*Clinic:* HealthCare Hospital, New York  
"""
)

if st.sidebar.button("Contact Doctor"):
    st.sidebar.write("Visit our contact page for more details!")

# File uploader for X-ray image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    try:
        img_size = (200, 200)  # Adjust size to match your model input
        if image.mode != "RGB":
            image = image.convert("RGB")  # Ensure image has 3 channels (RGB)
        image_resized = image.resize(img_size)
        image_array = img_to_array(image_resized) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_array)
        class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display the prediction
        st.markdown(f"<p style='font-weight: bold; color: #333; font-size: 2rem;'>Prediction: {predicted_class}</p>", unsafe_allow_html=True)

        # st.write(f"*Confidence:* {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer Section
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 50px;
    }
    </style>
    <div class="footer">
        <p>Developed by:</p>
        <p><b>Vanshika Mittal, Nikhil Chadha, Vanshika Narang, and Puru Sachdeva</b></p>
        <p>&copy; 2024 DeepLearning Team</p>
    </div>
    """,
    unsafe_allow_html=True,
)
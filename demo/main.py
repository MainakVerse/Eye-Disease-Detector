import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model import ImprovedTinyVGGModel
from utils import *
import datetime

# Configure page - MUST be the first Streamlit command
st.set_page_config(
    page_title="Ocular Eye Disease Classification",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_tailwind_css():
    # Apply Tailwind CSS styles via CDN with improved spacing
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #111827;
            color: #f3f4f6;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            border-radius: 0.5rem 0.5rem 0 0;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.1);
            border-bottom: 2px solid rgb(59, 130, 246);
        }
        .card {
            padding: 0.5rem;
            border-radius: 0.75rem;
            background-color: yellow;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 0.5rem;
            height: calc(100% - 1.5rem);
        }
        .uploadedFile {
            border: 1px dashed #ccc;
            border-radius: 8px;
            padding: 30px;
            margin: 20px 0;
            background-color: #f8fafc;
        }
        .prediction-card {
            border-radius: 0.75rem;
            padding: 2rem;
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 1.5rem;
        }
        .sidebar-card {
            background-color: white;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        }
        /* Column spacing fix */
        .row-container {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .column {
            flex: 1;
            min-width: 0;
        }
        /* Custom file uploader styling */
        .stFileUploader > div:first-child {
            width: 100%;
            padding: 0;
        }
        .stFileUploader > div:first-child > div:first-child {
            width: 100%;
        }
        /* Dark mode and contrast adjustments */
        h1, h2, h3, h4, h5 {
            color: #1e3a8a;
        }
        p {
            color: #374151;
        }
        /* Improve tab selection visibility */
        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.2);
            border-bottom: 3px solid rgb(59, 130, 246);
        }
    </style>
    """, unsafe_allow_html=True)

      # Main content
    st.sidebar.markdown("""
    <h1 class="text-3xl font-bold text-blue-700 mb-6">Ocular Eye Disease Classification</h1>
    """, unsafe_allow_html=True)

def main():
    apply_tailwind_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-card">
            <h2 class="text-xl font-bold text-gray-800 mb-4">Optician AI</h2>
            <p class="text-gray-600 mb-2">AI-powered eye disease detection tool</p>
            <hr class="my-4">
            <p class="text-sm text-gray-500">Device: CPU/GPU</p>
        </div>
        """, unsafe_allow_html=True)
        
       
    
  
    
    # Creating tabs - each with 4 columns width
    tab1, tab2, tab3 = st.tabs(["Detection", "About", "Optician AI"])
    
    # Detection Tab - Using custom row container for better spacing
    with tab1:
        # Use HTML for better control of spacing
        st.markdown("""
        <div class="row-container">
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Upload Eye Image</h2>
                    <p class="text-gray-600 mb-4">Upload a clear image of the eye for disease detection</p>
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Image Preview</h2>
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Now use regular columns for the content
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            # Setting device agnostic code
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load Trained Model
            MODEL_SAVE_PATH = "models/MultipleEyeDiseaseDetectModel.pth"
            model_info = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))
            
            # Instantiate Model
            model = ImprovedTinyVGGModel(
                input_shape=3,
                hidden_units=48,
                output_shape=6).to(device)
            
            # Define paths
            data_path = Path("demo/test_images/")
            
            
            
            uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="eyeimage")
            
        with col2:
            if uploaded_file is not None:
                # Save the uploaded image
                custom_image_path = data_path / uploaded_file.name
                with open(custom_image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Display the uploaded image with padding
                st.markdown('<div style="padding: 1rem 0;"></div>', unsafe_allow_html=True)
                image = Image.open(custom_image_path)
                st.image(image, caption='Uploaded Eye Image', use_column_width=True)
        
        with col3:
            if uploaded_file is not None:
                # Load and preprocess the image
                custom_image_transformed = load_and_preprocess_image(custom_image_path)
                
                # Load the model
                model.load_state_dict(model_info)
                model.eval()
                
                # Predict the label for the image
                class_names = np.array(['AMD', 'Cataract', 'Glaucoma', 'Myopia', 'Non-eye', 'Normal'])
                predicted_label, image_pred_probs = predict_image(model,
                                                                  custom_image_transformed,
                                                                  class_names)
                
                # Display prediction with modern styling
                confidence = image_pred_probs.max() * 100
                
                # Determine severity color based on confidence and condition
                if predicted_label[0] == "Normal":
                    severity_color = "green"
                elif confidence > 85:
                    severity_color = "red"
                elif confidence > 70:
                    severity_color = "orange"
                else:
                    severity_color = "blue"
                
                # Add top margin to align with image
                st.markdown('<div style="padding: 1rem 0;"></div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3 class="text-xl font-bold mb-4">Diagnosis:</h3>
                    <p class="text-{severity_color}-600 text-2xl font-bold mb-4">{predicted_label[0]}</p>
                    <div class="mt-6">
                        <p class="text-gray-700 mb-2">Confidence:</p>
                        <div class="w-full bg-gray-200 rounded-full h-4 mb-2">
                            <div class="bg-{severity_color}-600 h-4 rounded-full" style="width: {confidence:.2f}%"></div>
                        </div>
                        <p class="text-right text-sm text-gray-600">{confidence:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Update time for last prediction
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state['last_prediction_time'] = current_time
                
                
    
    # About Tab with improved spacing
    with tab2:
        st.markdown("""
        <div class="row-container">
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">About This Tool</h2>
                    <p class="text-gray-600">
                        This application uses deep learning to analyze eye images and detect various ocular diseases.
                        The model is trained to identify:
                    </p>
                    <ul class="list-disc pl-5 mt-3 text-gray-600">
                        <li>Age-related Macular Degeneration (AMD)</li>
                        <li>Cataract</li>
                        <li>Glaucoma</li>
                        <li>Myopia</li>
                        <li>Normal eyes</li>
                    </ul>
                    <p class="text-gray-600 mt-3">
                        Our model uses an improved TinyVGG architecture optimized for eye disease classification.
                    </p>
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">How It Works</h2>
                    <p class="text-gray-600">
                        The application processes uploaded eye images through the following steps:
                    </p>
                    <ol class="list-decimal pl-5 mt-3 text-gray-600">
                        <li>Image preprocessing (resizing, normalization)</li>
                        <li>Feature extraction through convolutional layers</li>
                        <li>Classification using fully connected layers</li>
                        <li>Probability calculation for each possible condition</li>
                    </ol>
                    <p class="text-gray-600 mt-3">
                        The highest probability class is presented as the diagnosis with a confidence score.
                    </p>
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Disclaimer</h2>
                    <p class="text-gray-600">
                        This tool is designed to assist in early detection but is not a replacement for professional medical advice.
                    </p>
                    <p class="text-gray-600 mt-3 font-semibold text-red-600">
                        Always consult with an ophthalmologist or eye care professional for proper diagnosis and treatment.
                    </p>
                    <p class="text-gray-600 mt-3">
                        The accuracy of this tool depends on image quality and proper positioning of the eye in the uploaded image.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Optician AI Tab with improved spacing
    with tab3:
        st.markdown("""
        <div class="row-container">
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Optician AI</h2>
                    <p class="text-gray-600">
                        Optician AI is an advanced platform for eye care professionals that leverages artificial intelligence to enhance diagnosis capabilities.
                    </p>
                    <p class="text-gray-600 mt-3">
                        Our mission is to make eye disease detection more accessible, accurate, and efficient through cutting-edge technology.
                    </p>
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Our Technology</h2>
                    <p class="text-gray-600">
                        The core of our system uses a custom neural network architecture specifically optimized for ocular disease detection.
                    </p>
                    <p class="text-gray-600 mt-3">
                        Key features:
                    </p>
                    <ul class="list-disc pl-5 mt-2 text-gray-600">
                        <li>High accuracy classification</li>
                        <li>Rapid processing time</li>
                        <li>Support for various image formats</li>
                        <li>Regular model updates</li>
                    </ul>
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Contact Us</h2>
                    <p class="text-gray-600">
                        For more information about Optician AI or to provide feedback, please contact our team.
                    </p>
                    <div class="mt-4">
                        <p class="text-gray-600"><span class="font-semibold">Email:</span> support@opticianai.com</p>
                        <p class="text-gray-600 mt-1"><span class="font-semibold">Website:</span> www.opticianai.com</p>
                    </div>
                    <p class="text-gray-600 mt-4 text-sm">
                        © 2025 Optician AI. All rights reserved.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Add footer
    st.markdown("""
    <footer class="p-4 mt-6 text-center text-gray-500 text-sm">
        <p>Developed with ❤️ by Mainak | 2025</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

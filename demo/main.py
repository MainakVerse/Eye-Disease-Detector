import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model import ImprovedTinyVGGModel
from utils import *
import streamlit.components.v1 as components

def apply_tailwind_css():
    # Apply Tailwind CSS styles via CDN
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
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
        .css-1544g2n.e1fqkh3o4 {
            padding: 2rem 1rem;
            border-radius: 0.5rem;
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        .uploadedFile {
            border: 1px dashed #ccc;
            border-radius: 5px;
            padding: 20px;
        }
        .prediction-card {
            border-radius: 0.5rem;
            padding: 1.5rem;
            background-color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 1rem;
        }
        .sidebar-card {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    apply_tailwind_css()
    
    # Configure page
    st.set_page_config(
        page_title="Ocular Eye Disease Classification",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-card">
            <h2 class="text-xl font-bold text-gray-800 mb-4">Optician AI</h2>
            <p class="text-gray-600 mb-2">AI-powered eye disease detection tool</p>
            <hr class="my-3">
            <p class="text-sm text-gray-500">Device: CPU/GPU</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction results will appear here when available
        st.markdown("""
        <div class="sidebar-card" id="prediction-results">
            <h3 class="text-lg font-semibold text-gray-800 mb-3">Prediction Results</h3>
            <p class="text-sm text-gray-500">Upload an image to see the prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-card">
            <h3 class="text-lg font-semibold text-gray-800 mb-3">About</h3>
            <p class="text-sm text-gray-600">This application uses a deep learning model to classify various eye diseases from images.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
    <h1 class="text-3xl font-bold text-blue-700 mb-6">Ocular Eye Disease Classification</h1>
    """, unsafe_allow_html=True)
    
    # Creating tabs - each with 4 columns width
    tab1, tab2, tab3 = st.tabs(["Detection", "About", "Optician AI"])
    
    # Detection Tab
    with tab1:
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Upload Eye Image</h2>
                <p class="text-gray-600 mb-4">Upload a clear image of the eye for disease detection</p>
            </div>
            """, unsafe_allow_html=True)
            
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
            
            # Image upload section
            st.markdown("""
            <div class="uploadedFile">
                <p class="text-center text-gray-500">Drag and drop your image here</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="eyeimage")
            
        with col2:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Image Preview</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Save the uploaded image
                custom_image_path = data_path / uploaded_file.name
                with open(custom_image_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Display the uploaded image
                image = Image.open(custom_image_path)
                st.image(image, caption='Uploaded Eye Image', use_column_width=True)
        
        with col3:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
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
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h3 class="text-xl font-bold mb-4">Diagnosis:</h3>
                    <p class="text-{severity_color}-600 text-2xl font-bold mb-2">{predicted_label[0]}</p>
                    <div class="mt-4">
                        <p class="text-gray-700 mb-1">Confidence:</p>
                        <div class="w-full bg-gray-200 rounded-full h-2.5">
                            <div class="bg-{severity_color}-600 h-2.5 rounded-full" style="width: {confidence:.2f}%"></div>
                        </div>
                        <p class="text-right text-sm text-gray-600">{confidence:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Update sidebar with prediction results
                st.sidebar.markdown(f"""
                <div class="sidebar-card" id="prediction-results">
                    <h3 class="text-lg font-semibold text-gray-800 mb-3">Prediction Results</h3>
                    <p class="font-bold text-{severity_color}-600">{predicted_label[0]}</p>
                    <p class="text-sm text-gray-700">Confidence: {confidence:.2f}%</p>
                    <p class="text-xs text-gray-500 mt-2">Last updated: {st.session_state.get('last_prediction_time', 'just now')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # About Tab
    with tab2:
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
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
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
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
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
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
            """, unsafe_allow_html=True)
    
    # Optician AI Tab
    with tab3:
        col1, col2, col3 = st.columns([4, 4, 4])
        
        with col1:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Optician AI</h2>
                <p class="text-gray-600">
                    Optician AI is an advanced platform for eye care professionals that leverages artificial intelligence to enhance diagnosis capabilities.
                </p>
                <p class="text-gray-600 mt-3">
                    Our mission is to make eye disease detection more accessible, accurate, and efficient through cutting-edge technology.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
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
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="p-4 bg-white rounded-lg shadow">
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
            """, unsafe_allow_html=True)

    # Add footer
    st.markdown("""
    <footer class="p-4 mt-6 text-center text-gray-500 text-sm">
        <p>Developed with ❤️ by Optician AI Team | 2025</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

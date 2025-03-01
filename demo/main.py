import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model import ImprovedTinyVGGModel
from utils import *
import datetime
import google.generativeai as genai

# Configure page - MUST be the first Streamlit command
st.set_page_config(
    page_title="Eye Disease Detector",
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
            gap: 3.5rem; /* Increase spacing between tabs */
            background-color: white;
            margin-bottom: 1.5rem;
            border-radius: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3.5rem; /* Increase tab height */
            min-width: 250px; /* Set a minimum width for wider tabs */
            white-space: pre-wrap;
            color: black;
            border-radius: 0.5rem 0.5rem 0 0;
            padding: 1rem 2rem; /* Increase padding for better spacing */
            font-weight: 700; /* Make text bolder */
            font-size: 4em; /* Increase font size */
            text-align: center;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(59, 130, 246, 0.2); /* Slightly brighter highlight */
            border-bottom: 3px solid rgb(59, 130, 246); /* Thicker bottom border */
            color: #ffffff;
        }

        @keyframes neonGlow {
            0% { box-shadow: 0 0 10px #ff00ff, 0 0 12px #ff00ff, 0 0 15px #ff00ff; }
            50% { box-shadow: 0 0 10px #00ff00, 0 0 12px #00ff00, 0 0 15px #00ff00; }
            100% { box-shadow: 0 0 10px #00ffff, 0 0 12px #00ffff, 0 0 15px #00ffff; }
        }

        .cardhome {
            padding: 1.5rem;
            border-radius: 0.75rem;
            background-color: white;
            margin: 0.5rem;
            text-align: center;
            color: #00ffcc;                  
            animation: neonGlow 3s infinite alternate;
            height: calc(100% - 1.5rem);
            transition: 0.3s ease-in-out;
        }

        .card {
            padding: 1.5rem;
            border-radius: 0.75rem;
            background-color: white;
            margin: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 0.5rem;
            height: calc(100% - 1.5rem);
        }
        .uploadedFile {
            border: 1px dashed #ccc;
            border-radius: 8px;
            padding: 30px;
            margin: 20px 0;
            color: black;
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
    tab1, tab2, tab3 = st.tabs(["DETECTION", "ABOUT", "OPTICIAN AI"])
    
    # Detection Tab - Using custom row container for better spacing
    with tab1:
        # Use HTML for better control of spacing
        st.markdown("""
        <div class="row-container">
            <div class="column">
                <div class="cardhome">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Upload Eye Image</h2>
                    <p class="text-gray-600 mb-4"></p>
                </div>
            </div>
            <div class="column">
                <div class="cardhome">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Image Preview</h2>
                </div>
            </div>
            <div class="column">
                <div class="cardhome">
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
        st.markdown('<h1 class="section-header">Optician AI Assistant</h1>', unsafe_allow_html=True)
    
        st.markdown("""
        <div class="info-card">
            <h3 style="color:#000000;">Ask the Optician AI</h3>
            <p>Get expert advice on eye health, vision problems, and eyewear recommendations. Our AI-powered Optician Assistant can answer your questions about eye care and vision management.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom CSS for the tablet-like response area and typewriter effect
        st.markdown("""
        <style>
            .info-card {
                background-color: #f0f7ff;
                border-radius: 10px;
                padding: 20px;
                border-left: 4px solid #3498db;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .tablet-response {
                background-color: #f7f9fc;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #e0e5ec;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
                margin-bottom: 20px;
                font-family: 'Courier New', monospace;
                max-height: 300px;
                overflow-y: auto;
            }
            
            /* Custom scrollbar for the tablet */
            .tablet-response::-webkit-scrollbar {
                width: 8px;
            }
            
            .tablet-response::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            
            .tablet-response::-webkit-scrollbar-thumb {
                background: #3498db;
                border-radius: 10px;
                color: #000000;
            }
            
            .typewriter-text {
                overflow: hidden;
                border-right: .15em solid #3498db;
                white-space: pre-wrap;
                margin: 0 auto;
                letter-spacing: .1em;
                color: #000000;
                animation: 
                    typing 3.5s steps(40, end),
                    blink-caret .75s step-end infinite;
            }
            
            @keyframes typing {
                from { max-width: 0 }
                to { max-width: 100% }
            }
            
            @keyframes blink-caret {
                from, to { border-color: transparent }
                50% { border-color: #3498db; }
            }
            
            .chat-message-user {
                background-color: #E8F4FD;
                padding: 10px 15px;
                border-radius: 18px 18px 18px 0;
                margin-bottom: 10px;
                display: inline-block;
                max-width: 80%;
                color: #000000;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            
            .chat-message-bot {
                background-color: #F0F7FF;
                padding: 10px 15px;
                color: #000000;
                border-radius: 18px 18px 0 18px;
                margin-bottom: 10px;
                margin-left: auto;
                display: inline-block;
                max-width: 80%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                border-left: 2px solid #3498db;
            }
            
            .chat-container {
                display: flex;
                flex-direction: column;
            }
            
            .user-container {
                display: flex;
                justify-content: flex-start;
                margin-bottom: 15px;
            }
            
            .bot-container {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 15px;
            }
            
            .question-button {
                background-color: #f8f9fa;
                border: 1px solid #e0e5ec;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: block;
                width: 100%;
                text-align: left;
            }
            
            .question-button:hover {
                background-color: #E8F4FD;
                border-left: 3px solid #3498db;
            }
            
            .stButton>button {
                background-color: #f8f9fa;
                border: 1px solid #e0e5ec;
                border-radius: 8px;
                padding: 10px;
                transition: all 0.3s ease;
                width: 100%;
                text-align: left;
            }
            
            .stButton>button:hover {
                background-color: #E8F4FD;
                border-left: 3px solid #3498db;
            }
            
            .ask-button {
                background-color: #3498db !important;
                color: white !important;
                font-weight: bold !important;
                border-radius: 8px !important;
                border: none !important;
                padding: 10px 15px !important;
                text-align: center !important;
                transition: all 0.3s ease !important;
            }
            
            .ask-button:hover {
                background-color: #2980b9 !important;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
        </style>
        """, unsafe_allow_html=True)
    
        # Display a relevant image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/10058/10058014.png", use_column_width=True)
        
        with col2:
            # Initialize chat history
            if "optician_chat_history" not in st.session_state:
                st.session_state.optician_chat_history = []
                
            # Initialize a session state for the selected question
            if "optician_selected_question" not in st.session_state:
                st.session_state.optician_selected_question = ""
                
            # Load API Key from Streamlit secrets
            try:
                GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
                if not GEMINI_API_KEY:
                    st.error("API key is missing! Add it to Streamlit secrets.")
            except:
                st.warning("To enable the Optician AI chatbot, please add your Gemini API key to Streamlit secrets.")
                GEMINI_API_KEY = None
                
            if GEMINI_API_KEY:
                # Configure Gemini API
                genai.configure(api_key=GEMINI_API_KEY)
                
                # Function to ask Gemini AI about eye health
                def ask_optician_ai(query):
                    prompt = f"""
                    You are an optician AI assistant specialized in eye health, vision problems, and eyewear recommendations. 
                    Answer only eye care-related queries with medically accurate information.
                    If a question is unrelated to eye care or optometry, politely inform the user that you can 
                    only answer eye health-related questions.
                    
                    Especially focus on these areas:
                    - Vision problems (myopia, hyperopia, astigmatism, presbyopia)
                    - Eye diseases (glaucoma, cataracts, AMD, diabetic retinopathy)
                    - Contact lenses and eyeglasses
                    - Vision correction options
                    - Eye health maintenance
                    - Digital eye strain
                    - Dry eye syndrome
                    - Children's vision
                    
                    **User's Question:** {query}
                    Provide a clear, concise, and accurate response about eye health and vision care.
                    """
                    model = genai.GenerativeModel("gemini-1.5-pro-latest")
                    response = model.generate_content(prompt)
                    
                    return response.text
                
                # User input - note that we're using the session state value as the default
                user_query = st.text_input("Ask your question about eye health:", 
                                          value=st.session_state.optician_selected_question,
                                          key="optician_ai_query")
                
                # After the user submits a question, clear the selected_question
                col1, col2 = st.columns([3, 1])
                with col2:
                    ask_button = st.button("Ask Optician AI", key="ask_optician", type="primary")
                    st.markdown("""
                    <style>
                        div[data-testid="stButton"] > button[kind="primary"] {
                            background-color: #3498db;
                            color: white;
                            font-weight: bold;
                            border-radius: 8px;
                            border: none;
                            padding: 8px 16px;
                            width: 100%;
                            text-align: center;
                        }
                        div[data-testid="stButton"] > button[kind="primary"]:hover {
                            background-color: #2980b9;
                            transform: translateY(-2px);
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                
                if ask_button:
                    if user_query:
                        with st.spinner("Optician AI is analyzing your question..."):
                            try:
                                # Get the response
                                response = ask_optician_ai(user_query)
                                # Add to chat history
                                st.session_state.optician_chat_history.append(("You", user_query))
                                st.session_state.optician_chat_history.append(("Optician AI", response))
                                # Clear the selected question after submission
                                st.session_state.optician_selected_question = ""
                            except Exception as e:
                                st.error(f"Error connecting to Gemini AI: {str(e)}")
            else:
                st.info("The Optician AI chatbot requires a Gemini API key to function.")
                user_query = st.text_input("Ask your question about eye health:", key="optician_ai_query", disabled=True)
                st.button("Ask Optician AI", disabled=True)
    
        # Display chat history in tablet-like response area
        if "optician_chat_history" in st.session_state and len(st.session_state.optician_chat_history) > 0:
            st.subheader("Conversation with Optician AI")
            
            # Create a tablet-like container for the conversation
            with st.container():
                st.markdown('<div class="tablet-response">', unsafe_allow_html=True)
                
                for i, (role, message) in enumerate(st.session_state.optician_chat_history):
                    if role == "You":
                        st.markdown(f'<div class="user-container"><div class="chat-message-user"><strong>👤 {role}:</strong> {message}</div></div>', unsafe_allow_html=True)
                    else:
                        # For the latest bot response, add the typewriter effect
                        if i == len(st.session_state.optician_chat_history) - 1 and role == "Optician AI":
                            st.markdown(f'<div class="bot-container"><div class="chat-message-bot"><strong>👁️ {role}:</strong> <span class="typewriter-text">{message}</span></div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="bot-container"><div class="chat-message-bot"><strong>👁️ {role}:</strong> {message}</div></div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
        # Add some common questions as examples
        st.markdown('<h3 class="section-header">Common Eye Health Questions</h3>', unsafe_allow_html=True)
        
        example_questions = [
            "What are the early signs of glaucoma?",
            "How often should I get my eyes checked?",
            "What causes dry eyes and how can I treat them?",
            "Are blue light glasses effective for digital eye strain?",
            "What's the difference between progressive and bifocal lenses?"
        ]
        
        # Create functions for handling button clicks
        def set_optician_question(question):
            st.session_state.optician_selected_question = question
        
        col1, col2 = st.columns(2)
        for i, question in enumerate(example_questions):
            if i % 2 == 0:
                with col1:
                    st.button(f"👁️ {question}", key=f"opt_q{i}", on_click=set_optician_question, args=(question,))
            else:
                with col2:
                    st.button(f"👁️ {question}", key=f"opt_q{i}", on_click=set_optician_question, args=(question,))
    
    # Add footer
    st.sidebar.markdown("""
    <footer class="p-4 mt-6 text-center text-gray-500 text-sm">
        <p>Developed with ❤️ by Mainak | 2025</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

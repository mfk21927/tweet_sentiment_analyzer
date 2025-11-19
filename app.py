import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle
import os
import time

# --- Configuration ---
# Must match the constants used during training
MAX_LEN = 50
MODEL_PATH = 'C:/Users/DELL/Desktop/project/sentiment_bilstm_model.h5'
TOKENIZER_PATH = 'C:/Users/DELL/Desktop/project/tokenizer.pickle'

# --- Helper Functions ---

# Function to load model and tokenizer
@st.cache_resource
def load_assets():
    """Loads the model and tokenizer, providing feedback if files are missing."""
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        # Load the Tokenizer
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except FileNotFoundError:
        st.error(f"Required files not found. Please ensure both '{MODEL_PATH}' and '{TOKENIZER_PATH}' exist. Run 'train_and_save.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during asset loading: {e}")
        st.stop()

# Function to preprocess text and make prediction
def predict_sentiment(text, model, tokenizer):
    """Tokenizes, pads, and predicts sentiment for the given text."""
    # 1. Tokenize
    sequence = tokenizer.texts_to_sequences([text])
    
    # 2. Pad
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    # Sigmoid output is the probability of class 1 (Positive)
    prob_positive = float(prediction)
    prob_negative = 1.0 - prob_positive
    
    return prob_negative, prob_positive

# --- Streamlit App UI and Logic ---

# Set a beautiful page configuration (Tailwind aesthetics)
st.set_page_config(
    page_title="Bi-LSTM Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Styling (using Markdown for simple CSS injection)
st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fb;
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #333333;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-box textarea {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .input-box textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        margin-top: 10px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .result-container {
        margin-top: 2rem;
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
    }
    .positive-border { border-left-color: #10b981; }
    .negative-border { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Load assets (Model and Tokenizer)
model, tokenizer = load_assets()

# --- Title and Description ---
st.markdown('<div class="main-title">Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicting Positive (1) or Negative (0) sentiment using a Bi-LSTM network.</div>', unsafe_allow_html=True)

# --- User Input ---
user_input = st.text_area(
    "Enter a piece of text (e.g., a tweet or review):",
    placeholder="Example: I am extremely happy with my new phone!",
    key="input_text",
    height=150
)
st.markdown('<div class="input-box"></div>', unsafe_allow_html=True)


# --- Prediction Button ---
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Show a loading spinner while predicting
        with st.spinner('Analyzing...'):
            time.sleep(0.5) # Simulate a small delay for better user experience
            
            # Get prediction probabilities
            prob_neg, prob_pos = predict_sentiment(user_input, model, tokenizer)
            
            # Determine the final sentiment result
            if prob_pos > 0.5:
                sentiment_result = "Positive"
                main_prob = prob_pos
                bar_class = "positive-border"
            else:
                sentiment_result = "Negative"
                main_prob = prob_neg
                bar_class = "negative-border"
                
            # --- Display Results ---
            
            # Result Card
            st.markdown(f'<div class="result-container {bar_class}">', unsafe_allow_html=True)
            st.subheader(f"Predicted Sentiment: {sentiment_result}")
            st.success(f"Confidence: {main_prob * 100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            
            # --- Probability Plot ---
            st.markdown("---")
            st.subheader("Probability Distribution")
            
            # Create a DataFrame for the bar chart
            chart_data = pd.DataFrame({
                'Class': ['Negative (0)', 'Positive (1)'],
                'Probability': [prob_neg, prob_pos]
            })
            
            # Use Streamlit's built-in bar chart
            # FIX: We changed 'color=['#ef4444', '#10b981']' to color='Class'.
            # This tells Streamlit/Altair to color the bars based on the categorical 'Class' column, 
            # resolving the length mismatch error.
            st.bar_chart(
                chart_data, 
                x='Class', 
                y='Probability',
                color='Class' # Use the 'Class' column to color the two bars distinctly
            )
            
            # Optional detailed probabilities
            col1, col2 = st.columns(2)
            col1.metric("Negative Probability", f"{prob_neg * 100:.2f}%", help="Probability of being class 0.")
            col2.metric("Positive Probability", f"{prob_pos * 100:.2f}%", help="Probability of being class 1.")

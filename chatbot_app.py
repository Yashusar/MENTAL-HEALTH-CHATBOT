import streamlit as st
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
from dotenv import load_dotenv
import google.generativeai as gen_ai

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Get the Gemini API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')


def get_response(user_message):
    # Check for emergency keywords
    emergency_keywords = ['suicide', 'self-harm', 'harm myself', 'end my life']
    if any(keyword in user_message.lower() for keyword in emergency_keywords):
        return "I'm really sorry you're feeling this way. Please reach out to a mental health professional or contact a helpline immediately."

    try:
        # Start a chat session and send user prompt to the Gemini API
        chat_session = model.start_chat(history=[])
        gemini_response = chat_session.send_message(user_message)

        return gemini_response.text.strip() if gemini_response else "Sorry, I couldn't process your request."
    except Exception as e:
        return f"An error occurred: {e}"


def analyze_sentiment(user_message):
    sentiment_score = sentiment_analyzer.polarity_scores(user_message)
    return sentiment_score['compound']


# Streamlit UI
st.title("CALM CONNECT")

st.write("Feel free to ask any questions or share your thoughts about mental health.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input from user
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get bot response
        bot_response = get_response(user_input)
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Bot:** {chat['content']}")

import os
import nltk
import ssl
import streamlit as st
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Setup
# -------------------------------
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(text):
    return text.lower()

# -------------------------------
# INTENTS
# -------------------------------
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": [
            "Hello! How can I assist you today?",
            "Hi there! I'm here to help you.",
            "Hey! What would you like to know?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Feel free to come back anytime."
        ]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": [
            "You're welcome! Happy to help.",
            "No problem at all!",
            "Glad I could assist you!"
        ]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you"],
        "responses": [
            "I am an AI chatbot built using Machine Learning techniques like TF-IDF and Logistic Regression.",
            "I can answer questions, assist you with basic queries, and demonstrate how ML-powered chatbots work."
        ]
    },
    {
        "tag": "budget",
        "patterns": [
            "How can I make a budget",
            "budget strategy",
            "create a budget",
            "financial budget",
            "money management",
            "how to save money",
            "expense planning"
        ],
        "responses": [
            "To create a good budget, start by tracking your income and expenses for at least a month. Categorize your spending into essentials like rent, food, and bills, and non-essentials like entertainment.",

            "A popular budgeting method is the 50/30/20 rule. Allocate 50% of your income to needs, 30% to wants, and 20% to savings or debt repayment. This helps maintain financial balance.",

            "Set clear financial goals before creating a budget. Then divide your income into categories such as essentials, savings, and discretionary spending. Review and adjust regularly."
        ]
    }
]

# -------------------------------
# Prepare training data
# -------------------------------
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(preprocess(pattern))
        tags.append(intent['tag'])

# -------------------------------
# Model
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

clf = LogisticRegression(max_iter=10000)
clf.fit(X, tags)

# -------------------------------
# Chatbot function
# -------------------------------
def chatbot(user_input):
    user_input = preprocess(user_input)
    X_test = vectorizer.transform([user_input])

    probs = clf.predict_proba(X_test)
    max_prob = max(probs[0])

    if max_prob < 0.25:
        return "🤖 Sorry, I didn't understand that. Please try rephrasing your question."

    tag = clf.predict(X_test)[0]

    for intent in intents:
        if intent['tag'] == tag:
            # Combine 2 responses → longer answers
            return " ".join(random.sample(intent['responses'], k=min(2, len(intent['responses']))))

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("🤖 AI Chatbot")
st.write("This chatbot uses **TF-IDF + Logistic Regression (ML Model)**")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = chatbot(user_input)

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
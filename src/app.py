import streamlit as st
import torch
import re
import numpy as np

from model import SentimentLSTM
from tokenizer import Tokenizer
from utils import pad_sequences

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="AI Movie Sentiment",
    page_icon="🎬",
    layout="centered"
)

# --------------------
# CUSTOM CSS (ANIMATIONS + COLORS)
# --------------------
st.markdown("""
<style>

/* Background animation */
body {
    background: linear-gradient(-45deg, #ff4ecd, #4facfe, #43e97b, #fa709a);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Moving title */
.moving-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #ff512f, #dd2476, #1fa2ff, #12d8fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: moveText 4s infinite alternate;
}

@keyframes moveText {
    from {letter-spacing: 1px;}
    to {letter-spacing: 6px;}
}

/* Subtitle glow */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: white;
    text-shadow: 0 0 10px #fff, 0 0 20px #ff4ecd;
}

/* Card */
.card {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(255,255,255,0.3);
    backdrop-filter: blur(12px);
    margin-top: 20px;
}

/* Button */
.stButton button {
    background: linear-gradient(90deg, #ff512f, #f09819);
    color: white;
    font-size: 20px;
    border-radius: 30px;
    padding: 0.6em 2em;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 20px #ff512f;
}

/* Confidence text pulse */
.pulse {
    animation: pulse 1.5s infinite;
    color: #00ffcc;
    font-size: 22px;
    font-weight: bold;
}

@keyframes pulse {
    0% {opacity: 0.5;}
    50% {opacity: 1;}
    100% {opacity: 0.5;}
}

</style>
""", unsafe_allow_html=True)

# --------------------
# Constants
# --------------------
MAX_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load Model & Tokenizer
# --------------------
tokenizer = Tokenizer()
tokenizer.word2idx = torch.load("../tokenizer_vocab.pth")

model = SentimentLSTM(vocab_size=len(tokenizer.word2idx))
model.load_state_dict(torch.load("../sentiment_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------
# Preprocessing
# --------------------
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# --------------------
# Prediction
# --------------------
def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.encode(text)
    seq = pad_sequences([seq], MAX_LEN)
    x = torch.tensor(seq, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        prob = model(x).item()

    sentiment = "Positive 😀" if prob >= 0.5 else "Negative 😞"
    return sentiment, prob

# --------------------
# UI CONTENT
# --------------------
st.markdown("<div class='moving-title'>🎬 AI Movie Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-Time Emotion Detection with Deep Learning 🚀</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

review = st.text_area(
    "💬 Enter your movie review",
    placeholder="This movie completely blew my mind with its emotions..."
)

st.markdown("</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze = st.button("✨ Analyze Emotion ✨")

# --------------------
# Output
# --------------------
if analyze:
    if review.strip() == "":
        st.warning("⚠️ Please type a review first.")
    else:
        sentiment, confidence = predict_sentiment(review)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if "Positive" in sentiment:
            st.success(f"🌈 **{sentiment}**")
        else:
            st.error(f"🔥 **{sentiment}**")

        st.markdown(f"<div class='pulse'>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
        st.progress(min(confidence, 1.0))

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><center>💎 Built with PyTorch • Streamlit • Love 💖</center>", unsafe_allow_html=True)

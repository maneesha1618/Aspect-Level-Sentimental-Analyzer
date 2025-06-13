import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import numpy as np

# Load tokenizer
with open("model/tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

# Load model
model = tf.keras.models.load_model("model/aspect_model.keras")

# Preprocessing function
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=26, padding='post')[:, :26]
    return padded

# Label map
label_map = {0: "negative", 1: "positive", 2: "neutral"}

# Streamlit UI
st.set_page_config(page_title="Aspect Sentiment Analyzer", layout="centered")
st.title("🧠 Aspect-Based Sentiment Analyzer")

aspect_text = st.text_area("Enter an aspect or sentence:", "")

if st.button("Analyze Sentiment"):
    if aspect_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_seq = preprocess_text(aspect_text)
        pred = model.predict(input_seq)
        sentiment = label_map[np.argmax(pred)]
        
        st.success(f"**Sentiment:** {sentiment.capitalize()}")
        st.json({"aspect": aspect_text, "sentiment": sentiment})

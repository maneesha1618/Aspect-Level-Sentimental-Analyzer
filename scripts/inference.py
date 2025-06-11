# scripts/inference.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load tokenizer
with open("model/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(json.load(f))

model = load_model("model/aspect_model.keras")
label_map = {0: "negative", 1: "positive", 2: "neutral"}

def predict_aspect_sentiment(aspect):
    seq = tokenizer.texts_to_sequences([aspect])
    padded = pad_sequences(seq, maxlen=26, padding='post')
    pred = model.predict(padded)
    label = label_map[np.argmax(pred)]
    return label

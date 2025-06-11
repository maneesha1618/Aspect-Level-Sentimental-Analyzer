from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    aspect = data.get("aspect", "")
    
    if not aspect:
        return jsonify({"error": "Aspect text missing"}), 400
    
    input_seq = preprocess_text(aspect)
    pred = model.predict(input_seq)
    sentiment = label_map[pred.argmax()]
    
    return jsonify({"aspect": aspect, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

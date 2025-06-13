import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for even fewer logs
import tensorflow as tf


# Load SpaCy and model
nlp = spacy.load("en_core_web_sm")
with open("model/tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())
model = tf.keras.models.load_model("model/aspect_model.keras")
MAXLEN = model.input_shape[1]

label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
color_map = {"Negative": "#e57979", "Positive": "#76eca1", "Neutral": "#aba"}
emoji_map = {"Negative": "😠", "Positive": "😃", "Neutral": "😐"}

# ---------- Helper Functions ----------
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
    return padded

def extract_aspects(text):
    doc = nlp(text)
    aspects = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text.strip()) > 1:
            aspects.add(chunk.text.strip().lower())
    return list(aspects)

def analyze_aspects(aspect_list):
    results = []
    for aspect in aspect_list:
        padded_input = preprocess_text(aspect)
        pred = model.predict(padded_input, verbose=0)[0]
        idx = np.argmax(pred)
        sentiment = label_map[idx]
        confidence = float(pred[idx])
        results.append({
            "Aspect": aspect,
            "Sentiment": sentiment,
            "Confidence": confidence
        })
    return results

def render_sentiment_charts(df):
    counts = df['Sentiment'].value_counts().reindex(["Positive", "Negative", "Neutral"], fill_value=0)
    
    st.subheader("📊 Sentiment Distribution")

    fig1, ax1 = plt.subplots()
    ax1.bar(counts.index, counts.values, color=[color_map[k] for k in counts.index])
    ax1.set_ylabel("Count", color="white")
    ax1.set_xlabel("Sentiment", color="white")
    ax1.tick_params(colors='white')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=[color_map[k] for k in counts.index])
    ax2.axis('equal')
    st.pyplot(fig2)

def generate_confusion_matrix(df):
    if "TrueSentiment" in df.columns:
        label_map_inv = {"Negative": 0, "Positive": 1, "Neutral": 2}
        true = df["TrueSentiment"].map(label_map_inv)
        pred = df["Sentiment"].map(label_map_inv)
        if true.isnull().any() or pred.isnull().any():
            return
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=label_map.values(),
                    yticklabels=label_map.values())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.subheader("📉 Confusion Matrix")
        st.pyplot(fig)

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Aspect Sentiment Analyzer", layout="centered")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #141E30, #243B55);
            color: #ffffff;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.03);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        }
        .aspect-box {
            background-color: rgba(255,255,255,0.07);
            padding: 10px;
            margin-bottom: 12px;
            border-radius: 10px;
        }
        h1, h2, h3, h4, h5, h6, .stTextInput>div>div>input {
            color: #ffffff !important;
        }
        .stButton>button {
            background-color: #4c5fd5;
            color: white;
            border-radius: 10px;
            padding: 8px 20px;
        }
        .stButton>button:hover {
            background-color: #3b4ab5;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📁 Project Info")
st.sidebar.markdown("""
- **Project**: Aspect Sentiment Analyzer  
- **Model**: LSTM + Attention  
- **Language**: English  
- **Built With**: TensorFlow, Spacy, Streamlit  
""")
st.sidebar.success("👨‍💻 Developed by Maneesha M")

# Main Title
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🧠 Aspect-Based Sentiment Analyzer")

# Review input
st.markdown("Enter a review or upload CSV with reviews. The app will detect aspects and predict sentiment.")

text = st.text_area("✍️ Input Review")

if st.button("🔍 Analyze Review"):
    if not text.strip():
        st.warning("⚠️ Please enter review text.")
    else:
        aspects = extract_aspects(text)
        if not aspects:
            st.info("🤖 No aspects found.")
        else:
            results = analyze_aspects(aspects)
            df = pd.DataFrame(results)
            st.subheader("🧾 Detailed Results")
            for r in results:
                st.markdown(f"""
                    <div class="aspect-box">
                        <strong>{r['Aspect'].capitalize()}</strong><br>
                        Sentiment: <b style="color:{color_map[r['Sentiment']]};">{emoji_map[r['Sentiment']]} {r['Sentiment']}</b><br>
                        Confidence: {r['Confidence']:.2f}
                    </div>
                """, unsafe_allow_html=True)
            render_sentiment_charts(df)

# CSV upload
st.subheader("📤 Upload CSV for Bulk Analysis")
uploaded_file = st.file_uploader("Upload a CSV with a 'Review' column", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Review' not in df.columns:
        st.error("CSV must have a 'Review' column.")
    else:
        all_data = []
        for review in df['Review'].fillna(""):
            aspects = extract_aspects(review)
            results = analyze_aspects(aspects)
            for r in results:
                r['Review'] = review
                all_data.append(r)

        full_df = pd.DataFrame(all_data)
        st.dataframe(full_df, use_container_width=True)

        render_sentiment_charts(full_df)
        generate_confusion_matrix(full_df)

        csv_buffer = io.StringIO()
        full_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="aspect_sentiment_results.csv",
            mime="text/csv"
        )

st.markdown("</div>", unsafe_allow_html=True)


# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import json

# # Load tokenizer and model
# with open("model/tokenizer.json", "r") as f:
#     tokenizer = tokenizer_from_json(f.read())

# model = tf.keras.models.load_model("model/aspect_model.keras")
# MAXLEN = model.input_shape[1]

# # Label maps
# label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
# color_map = {"Negative": "red", "Positive": "green", "Neutral": "gray"}
# emoji_map = {"Negative": "😠", "Positive": "😃", "Neutral": "😐"}

# # Preprocess
# def preprocess_text(text):
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
#     return padded

# # Set page config and style
# st.set_page_config(page_title="Aspect Sentiment Analyzer", layout="centered")

# st.markdown("""
#     <style>
#         .main {
#             background: linear-gradient(to bottom right, #f3f9fa, #c7cfe2);
#             padding: 20px;
#             border-radius: 10px;
#         }
#         .big-text {
#             font-size: 36px !important;
#             font-weight: bold;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar
# st.sidebar.title("📁 Project Info")
# st.sidebar.markdown("""
# - **Project**: Aspect Sentiment Analyzer  
# - **Model**: LSTM + Attention  
# - **Language**: English  
# - **Built With**: TensorFlow, Streamlit  
# """)
# st.sidebar.success("👨‍💻 Developed by Maneesha M")

# # Title
# st.markdown("<div class='main'>", unsafe_allow_html=True)
# st.title("Aspect-Based Sentiment Analyzer")
# st.markdown("Enter an *aspect* or short *review phrase* to analyze sentiment.")
# st.markdown("---")

# # User Input
# user_input = st.text_area("✍️ Input Text", placeholder="e.g., battery backup, customer service")

# # Storage for previous results
# if "history" not in st.session_state:
#     st.session_state.history = []

# # On Prediction
# if st.button("🔍 Analyze"):
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text.")
#     else:
#         input_seq = preprocess_text(user_input)
#         pred = model.predict(input_seq)[0]
#         idx = np.argmax(pred)
#         sentiment = label_map[idx]
#         emoji = emoji_map[sentiment]
#         confidence = pred[idx]

#         # Show sentiment
#         st.markdown(f"<p class='big-text'>{emoji} {sentiment}</p>", unsafe_allow_html=True)
#         st.markdown("### 🔎 Confidence Scores")
#         score_dict = {label_map[i]: float(pred[i]) for i in range(3)}
#         st.json(score_dict)

#         # Bar Chart
#         fig, ax = plt.subplots()
#         ax.bar(score_dict.keys(), score_dict.values(), color=[color_map[k] for k in score_dict.keys()])
#         ax.set_title("Sentiment Confidence")
#         ax.set_ylabel("Probability")
#         st.pyplot(fig)

#         # Pie Chart
#         fig2, ax2 = plt.subplots()
#         ax2.pie(score_dict.values(), labels=score_dict.keys(), autopct='%1.1f%%',
#                 colors=[color_map[k] for k in score_dict.keys()], startangle=90)
#         ax2.axis('equal')
#         st.pyplot(fig2)

#         # Save to session
#         st.session_state.history.append({
#             "Aspect": user_input.strip(),
#             "Sentiment": sentiment,
#             "Confidence": round(confidence, 3)
#         })

# # Show history
# if st.session_state.history:
#     st.markdown("---")
#     st.markdown("### 📋 Prediction History")
#     history_df = pd.DataFrame(st.session_state.history[::-1])  # Most recent first
#     st.dataframe(history_df, use_container_width=True)

# st.markdown("</div>", unsafe_allow_html=True)


# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# import numpy as np
# import json
# import matplotlib.pyplot as plt

# # Load tokenizer
# with open("model/tokenizer.json", "r") as f:
#     tokenizer = tokenizer_from_json(f.read())

# # Load model
# model = tf.keras.models.load_model("model/aspect_model.keras")
# MAXLEN = model.input_shape[1]

# # Label map
# label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
# label_colors = {"Negative": "red", "Positive": "green", "Neutral": "gray"}

# # Preprocessing function
# def preprocess_text(text):
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')
#     return padded

# # Title and layout
# st.set_page_config(page_title="Aspect Sentiment Analyzer", layout="centered")
# st.title("🧠 Aspect-Based Sentiment Analyzer")
# st.markdown("Analyze the sentiment of any **aspect or phrase** using a trained deep learning model.")
# st.markdown("---")

# # Input text
# aspect_text = st.text_area("✍️ Enter an aspect or sentence:", placeholder="e.g., battery life, camera quality, performance")

# # Button
# if st.button("🔍 Analyze Sentiment"):
#     if aspect_text.strip() == "":
#         st.warning("⚠️ Please enter some text.")
#     else:
#         try:
#             input_seq = preprocess_text(aspect_text)
#             pred = model.predict(input_seq)[0]
#             sentiment_idx = int(np.argmax(pred))
#             sentiment = label_map[sentiment_idx]

#             # Display result
#             st.success(f"✅ **Predicted Sentiment:** {sentiment}")
#             st.write("### 🔢 Confidence Scores")

#             score_dict = {label_map[i]: float(pred[i]) for i in range(len(pred))}
#             st.json({"aspect": aspect_text.strip(), "sentiment": sentiment, "scores": score_dict})

#             # Visualization
#             fig, ax = plt.subplots()
#             ax.bar(score_dict.keys(), score_dict.values(), color=[label_colors[l] for l in score_dict.keys()])
#             ax.set_ylabel("Confidence")
#             ax.set_title("Sentiment Confidence Scores")
#             st.pyplot(fig)

#         except Exception as e:
#             st.error(f"❌ Prediction failed: {str(e)}")

# # Footer
# st.markdown("---")
# st.markdown("📊 Built with TensorFlow, Streamlit & LSTM + Attention model")

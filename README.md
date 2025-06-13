Here’s a clean and informative `README.md` file you can include in your project root. It outlines the purpose, setup steps, and usage clearly:

---

### 📄 `README.md`

```markdown
# 🧠 Aspect-Level Sentiment Analysis with RNN + Attention

This project performs **aspect-level sentiment analysis** using an RNN-based architecture with attention and multi-head attention layers. It extracts aspects from product reviews and classifies their sentiment as **positive**, **negative**, or **neutral**.

---

## 📁 Project Structure

```

aspect\_sentiment/
│
├── data/
│   └── AllProductReviews.csv           # Original dataset
│
├── model/
│   ├── aspect\_model.keras              # Trained Keras model
│   └── tokenizer.json                  # Tokenizer used for inference
│
├── scripts/
│   ├── preprocess.py                   # Cleans reviews, extracts aspects, and runs sentiment analysis
│   ├── train.py                        # Trains the RNN model
│   ├── inference.py                    # Inference logic for predictions
│
├── app/
│   └── app.py                          # Flask API to expose model
│
├── requirements.txt                    # Dependencies
└── README.md                           # Project documentation

````

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aspect-sentiment-rnn.git
cd aspect-sentiment-rnn
````

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the Project

### 1. Preprocess the Data

```bash
python scripts/preprocess.py
```

This will clean the review texts, extract aspects, and assign sentiment labels to each aspect. Output: `data/train.csv`.

### 2. Train the Model

```bash
python scripts/train.py
```

This will train an RNN + Attention model and save the model and tokenizer to the `model/` directory.

### 3. Start the Flask API Server

```bash
python app/app.py
```

The API will be running at `http://127.0.0.1:5000/`.

---

### 4. Start the Streamlit API Server
```bash
streamlit run app.py
```


## 🔍 API Usage

### Endpoint

**POST** `/predict`

**Request Body (JSON):**

```json
{
  "aspect": "battery"
}
```

**Response:**

```json
{
  "aspect": "battery",
  "sentiment": "positive"
}
```

Test with `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"aspect": "battery"}'
```

---

## 📊 Model Architecture

* **Embedding Layer** (trainable)
* **Bidirectional LSTM** (128 units)
* **Multi-Head Attention** (4 heads, 64-d keys)
* **Global Max + Average Pooling**
* **Dense + LeakyReLU + BatchNorm + Dropout**
* **Softmax Output for 3 classes**

---

## ✍️ Author

* Maneesha M. — AI/ML Engineer
* Trained using TensorFlow, Spacy, VADER, and Flask.

---

## 📌 Notes

* This project focuses on the sentiment **of specific aspects**, not entire reviews.
* Fine-tuning or using domain-specific aspect lists could improve accuracy further.

---

## 📦 Deployment (Coming Soon)

* Dockerization
* Serverless option (AWS Lambda + API Gateway)

```

---

Let me know if you’d like:
- The Dockerfile + deployment instructions (to make it production-ready).
- A GitHub-style repository template zip.
- A UI frontend using React to test the model online.

Would you like to proceed with Docker deployment next?
```



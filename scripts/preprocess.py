# # scripts/preprocess.py

# import pandas as pd
# import spacy
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import json

# nlp = spacy.load("en_core_web_sm")
# nltk.download("vader_lexicon")
# sid = SentimentIntensityAnalyzer()

# def clean_text(text):
#     from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#     stop_words = set(ENGLISH_STOP_WORDS)
#     doc = nlp(text.lower())
#     tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
#     return " ".join(tokens)

# def extract_aspects_loose(review):
#     doc = nlp(review)
#     aspects = set()
#     for chunk in doc.noun_chunks:
#         if len(chunk.text.strip()) > 1:
#             aspects.add(chunk.text.strip().lower())
#     for token in doc:
#         if token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
#             aspects.add(f"{token.text} {token.head.text}".lower())
#         if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
#             aspects.add(f"{token.text} {token.head.text}".lower())
#     return [a for a in aspects if a not in {"the", "is", "a", "an", "and", "this", "that", "with", "in", "on", "for", "of", "to", "it", "be", "by"}]

# def analyze_aspect_sentiment(review, aspects):
#     sentiment_dict = {}
#     for aspect in aspects:
#         relevant = [s for s in review.split('.') if aspect in s]
#         combined = ' '.join(relevant)
#         score = sid.polarity_scores(combined)['compound'] if combined else 0
#         if score >= 0.05:
#             sentiment_dict[aspect] = "positive"
#         elif score <= -0.05:
#             sentiment_dict[aspect] = "negative"
#         else:
#             sentiment_dict[aspect] = "neutral"
#     return sentiment_dict

# def run_pipeline(csv_path):
#     df = pd.read_csv(csv_path)
#     df['cleaned_review'] = df['ReviewBody'].fillna("").apply(clean_text)
#     df['aspects'] = df['cleaned_review'].apply(extract_aspects_loose)
#     df['aspect_sentiment'] = df.apply(lambda x: analyze_aspect_sentiment(x['cleaned_review'], x['aspects']), axis=1)

#     train_data = []
#     for d in df['aspect_sentiment']:
#         for asp, sent in d.items():
#             train_data.append((asp, sent))

#     train_df = pd.DataFrame(train_data, columns=['Aspect', 'Sentiment'])
#     train_df.to_csv("data/train.csv", index=False)
#     print("✅ Preprocessing complete. Saved as data/train.csv")

# if __name__ == "__main__":
#     run_pipeline("data/AllProductReviews.csv")


import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import os

nltk.download("vader_lexicon")
nltk.download("stopwords")

nlp = spacy.load("en_core_web_sm")
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return " ".join(tokens)

def extract_aspects_loose(review):
    doc = nlp(review)
    aspects = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text.strip()) > 1:
            aspects.add(chunk.text.strip().lower())
    for token in doc:
        if token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
            aspects.add(f"{token.text} {token.head.text}".lower())
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            aspects.add(f"{token.text} {token.head.text}".lower())
    return [a for a in aspects if a not in stop_words]

def analyze_aspect_sentiment(review, aspects):
    sentiment_dict = {}
    for aspect in aspects:
        relevant = [s for s in review.split('.') if aspect in s]
        combined = ' '.join(relevant)
        score = sid.polarity_scores(combined)['compound'] if combined else 0
        if score >= 0.05:
            sentiment_dict[aspect] = "positive"
        elif score <= -0.05:
            sentiment_dict[aspect] = "negative"
        else:
            sentiment_dict[aspect] = "neutral"
    return sentiment_dict

def run_pipeline(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    if 'ReviewBody' not in df.columns:
        print("❌ 'ReviewBody' column missing in CSV.")
        return

    df['cleaned_review'] = df['ReviewBody'].fillna("").apply(clean_text)
    df['aspects'] = df['cleaned_review'].apply(extract_aspects_loose)
    df['aspect_sentiment'] = df.apply(lambda x: analyze_aspect_sentiment(x['cleaned_review'], x['aspects']), axis=1)

    train_data = []
    for d in df['aspect_sentiment']:
        for asp, sent in d.items():
            train_data.append((asp.strip(), sent.strip()))

    train_df = pd.DataFrame(train_data, columns=['Aspect', 'Sentiment'])
    train_df.to_csv("data/train.csv", index=False)
    print("✅ Preprocessing complete. Saved as data/train.csv")

if __name__ == "__main__":
    run_pipeline("data/AllProductReviews.csv")

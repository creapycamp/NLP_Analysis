import pandas as pd
import numpy as np
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.decomposition import NMF

import gradio as gr
df = pd.read_csv(r"D:\Fundamentals of NLP\Reviews.csv")

# Use only important columns
df = df[['Review Text', 'Rating', 'Recommended IND']]
df.dropna(inplace=True)

df.head(10)
def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Rating'].apply(label_sentiment)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()   # ← simple tokenization

    tokens = [lemmatizer.lemmatize(word)
              for word in tokens
              if word not in stop_words]

    return " ".join(tokens)

df['Clean_Text'] = df['Review Text'].apply(preprocess)

print("Before:", df['Review Text'].iloc[0])
print("After:", df['Clean_Text'].iloc[0])

# Bag of Words
bow_vectorizer = CountVectorizer(max_features=5000)
X_bow = bow_vectorizer.fit_transform(df['Clean_Text'])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['Clean_Text'])

y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

def label_intent(text):
    text = text.lower()

    if "refund" in text or "return" in text:
        return "Refund Request"
    elif "late" in text or "delivery" in text or "shipping" in text:
        return "Delivery Issue"
    elif "bad" in text or "poor" in text or "damaged" in text:
        return "Complaint"
    else:
        return "General Query"

df['Intent'] = df['Review Text'].apply(label_intent)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_tfidf, df['Intent'], test_size=0.2, random_state=42
)

intent_model = LogisticRegression(max_iter=300)
intent_model.fit(X_train_i, y_train_i)

intent_pred = intent_model.predict(X_test_i)

print(classification_report(y_test_i, intent_pred))

nmf_model = NMF(n_components=3, random_state=42)
nmf_model.fit(X_tfidf)

feature_names = tfidf_vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"\nTopic {topic_idx+1}:")
    print([feature_names[i] for i in topic.argsort()[-10:]])

def predict_review(review):
    clean = preprocess(review)
    vector = tfidf_vectorizer.transform([clean])

    sentiment = model.predict(vector)[0]
    intent = intent_model.predict(vector)[0]

    topic = nmf_model.transform(vector)
    topic_id = topic.argmax() + 1

    return sentiment, intent, f"Topic {topic_id}"

interface = gr.Interface(
    fn=predict_review,
    inputs="text",
    outputs=["text", "text", "text"],
    title="Customer Review Intelligence System"
)

interface.launch()


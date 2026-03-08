import pandas as pd
import numpy as np
import re
import nltk
import gradio as gr

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import NMF

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(r"Data\Reviews.csv")

df = df[['Review Text', 'Rating', 'Recommended IND']]
df.dropna(inplace=True)

# -----------------------------
# Sentiment Labeling
# -----------------------------
def label_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Rating'].apply(label_sentiment)

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

df['Clean_Text'] = df['Review Text'].apply(preprocess)

# -----------------------------
# TF-IDF Feature Extraction
# -----------------------------
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = tfidf_vectorizer.fit_transform(df['Clean_Text'])
y = df['Sentiment']

# -----------------------------
# Train Sentiment Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sentiment_model = LogisticRegression(
    max_iter=500,
    class_weight="balanced"
)

sentiment_model.fit(X_train, y_train)

y_pred = sentiment_model.predict(X_test)

print("\nSentiment Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Intent Labeling
# -----------------------------
def label_intent(text):
    text = text.lower()

    refund_words = ["refund", "return", "money back"]
    delivery_words = ["delivery", "shipping", "delay", "late"]
    complaint_words = ["bad", "poor", "damaged", "defective", "broken"]

    if any(word in text for word in refund_words):
        return "Refund Request"

    elif any(word in text for word in delivery_words):
        return "Delivery Issue"

    elif any(word in text for word in complaint_words):
        return "Complaint"

    else:
        return "General Query"

df['Intent'] = df['Review Text'].apply(label_intent)

# -----------------------------
# Train Intent Model
# -----------------------------
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X, df['Intent'], test_size=0.2, random_state=42
)

intent_model = LogisticRegression(max_iter=500)

intent_model.fit(X_train_i, y_train_i)

intent_pred = intent_model.predict(X_test_i)

print("\nIntent Classification Report:\n")
print(classification_report(y_test_i, intent_pred))

# -----------------------------
# Topic Modeling (NMF)
# -----------------------------
nmf_model = NMF(n_components=3, random_state=42)
nmf_model.fit(X)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nTopics:\n")

for topic_idx, topic in enumerate(nmf_model.components_):
    print(f"Topic {topic_idx + 1}:")
    print([feature_names[i] for i in topic.argsort()[-10:]])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_review(review):

    clean = preprocess(review)

    vector = tfidf_vectorizer.transform([clean])

    sentiment = sentiment_model.predict(vector)[0]

    intent = intent_model.predict(vector)[0]

    topic = nmf_model.transform(vector)
    topic_id = topic.argmax() + 1

    return sentiment, intent, f"Topic {topic_id}"

# -----------------------------
# Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter customer review here..."
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Intent"),
        gr.Textbox(label="Topic")
    ],
    title="Customer Review Intelligence System",
    description="Analyze customer reviews using NLP"
)

interface.launch()
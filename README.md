# Customer Review Intelligence System

An NLP-based system that analyzes customer reviews to extract meaningful insights.  
The system performs **Sentiment Analysis**, **Query Classification**, and **Topic Modeling** to understand customer feedback.

A simple interactive interface is built using **Gradio** for real-time prediction.

---

## Project Features

• Sentiment Classification (Positive, Neutral, Negative)  
• Customer Query Classification (Complaint, Delivery Issue, General Query, Refund Request)  
• Topic Modeling using NMF  
• Interactive Web Interface using Gradio  
• NLP preprocessing with NLTK  
• Machine Learning models using Scikit-learn  

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Gradio
- MLflow (for experiment tracking)

---

## Project Workflow

1. Data preprocessing
2. Text cleaning and tokenization
3. Stopword removal and lemmatization
4. Feature extraction using TF-IDF / CountVectorizer
5. Sentiment classification using Logistic Regression
6. Query classification using Logistic Regression
7. Topic modeling using NMF
8. Deployment using Gradio interface

---

## Sentiment Classification Performance

| Class | Precision | Recall | F1-score |
|------|-----------|--------|---------|
| Negative | 0.56 | 0.42 | 0.48 |
| Neutral | 0.51 | 0.23 | 0.32 |
| Positive | 0.87 | 0.97 | 0.92 |

**Overall Accuracy:** 82%

### Confusion Matrix


[[ 193 75 189]
[ 116 137 335]
[ 33 57 3394]]


---

## Query Classification Performance

| Class | Precision | Recall | F1-score |
|------|-----------|--------|---------|
| Complaint | 0.98 | 0.44 | 0.61 |
| Delivery Issue | 1.00 | 0.08 | 0.14 |
| General Query | 0.96 | 1.00 | 0.98 |
| Refund Request | 1.00 | 0.90 | 0.94 |

**Overall Accuracy:** 96%

---

## Topic Modeling Results

Topic modeling is performed using **Non-negative Matrix Factorization (NMF)**.

### Topic 1

run, large, ordered, fit, would, im, like, small, top, size


### Topic 2

im, slip, fabric, wear, fit, perfect, love, flattering, beautiful, dress


### Topic 3

wear, sweater, perfect, fit, soft, comfortable, jean, color, great, love


These topics represent common themes found in customer reviews.

---

## Gradio Interface

The system includes an interactive interface where users can enter a customer review and get:

• Sentiment prediction  
• Query type classification  
• Topic identification  

Example Input:


very good quality


Example Output:


Sentiment: Positive
Query Type: General Query
Topic: Topic 3


---

## Installation

Clone the repository:

```bash
git clone https://github.com/creapycamp/NLP_Analysis.git
cd NLP_Analysis

Install dependencies:

pip install -r requirements.txt

Download required NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Run the Application
python app.py

The Gradio interface will open in your browser.

Future Improvements

• Improve class balance for better sentiment prediction
• Add deep learning models (LSTM / Transformers)
• Deploy using Docker
• Add real-time dashboard for analytics

Author
ROMAN HUSSAIN

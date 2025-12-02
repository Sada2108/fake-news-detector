# Fake News Detector (Indian News)

An AI‑powered Fake News Detector for Indian news articles and headlines. It uses both classical machine learning models and a BERT‑based transformer to classify news as **FAKE** or **REAL**, with an interactive Streamlit web app on top.

## Features

- Detects fake or real news for Indian headlines and articles using trained ML and BERT models.
- Uses TF‑IDF + classical models (Logistic Regression, Random Forest, SVM) and a fine‑tuned DistilBERT model for better accuracy.
- Streamlit interface where users can paste news text and instantly see predictions.
- SQLite memory: stores user corrections so the system can remember past feedback and improve over time.

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, NumPy, scikit‑learn, PyTorch, transformers, Streamlit, sqlite3.
- **Models:** TF‑IDF + classical classifiers, DistilBERT for sequence classification.

## Project Workflow

1. Load and merge multiple Indian news datasets (fake and real) using pandas, clean them, and create a combined text column.
2. Convert text to numerical features with TF‑IDF / BERT tokenization and split into train–test sets.
3. Train and evaluate classical ML models and a DistilBERT model, then save the best models and vectorizers.
4. Build a Streamlit app that loads the saved models, takes user input, and displays FAKE/REAL predictions.
5. Store user‑corrected labels in a SQLite database so repeated news items use the corrected label instead of the raw model output.

## How to Run

```bash
# 1. Install dependencies (example)
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py
```


Open the local URL Streamlit prints in your browser and test the fake news detector with your own headlines.

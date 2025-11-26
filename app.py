import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import sqlite3

DB_PATH = "corrections.db"

def init_sqlite():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            headline TEXT,
            article TEXT,
            label TEXT,
            PRIMARY KEY (headline, article)
        )
    """)
    conn.commit()
    conn.close()

def save_correction_sqlite(headline, article, label):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO corrections (headline, article, label) VALUES (?, ?, ?)
    """, (headline, article, label))
    conn.commit()
    conn.close()

def get_label_sqlite(headline, article):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT label FROM corrections WHERE headline=? AND article=?
    """, (headline, article))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

# --- INIT DB ---
init_sqlite()

# --- Load classic ML model and vectorizer ---
ml_model = joblib.load("best_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# --- Load fine-tuned BERT model ---
device = torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
bert_model.load_state_dict(torch.load("bert_fakenews.pt", map_location=device))
bert_model.to(device)
bert_model.eval()

st.title("Intelligent Fake News Detector")

headline = st.text_input("Enter headline:")
article = st.text_area("Paste a news article (optional):")
user_text = headline + " " + article

# -------- Memory/Correction and Prediction --------
user_label = get_label_sqlite(headline, article)
if user_label is not None:
    st.success(f"User-verified label: This news is **{user_label}** (from database, overrides model).")
else:
    prediction = None
    # ML Prediction
    if st.button("Predict (ML)"):
        features = vectorizer.transform([user_text])
        prediction = ml_model.predict(features)[0]
        st.write(f"Classic ML Prediction: This news is likely to be **{prediction}**.")
    # BERT Prediction
    if st.button("Predict with BERT"):
        inputs = tokenizer(user_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            pred = torch.argmax(outputs.logits, axis=1).item()
            prediction = "FAKE" if pred == 0 else "REAL"
        st.write(f"BERT Prediction: This news is likely to be **{prediction}**.")

    # Correction UI (always available after prediction)
    if prediction:
        st.info("Correct the prediction if needed:")
        if st.button("Mark as FAKE"):
            save_correction_sqlite(headline, article, "FAKE")
            st.success("Your correction was saved! Future answers will use this label.")
        if st.button("Mark as REAL"):
            save_correction_sqlite(headline, article, "REAL")
            st.success("Your correction was saved! Future answers will use this label.")

# --------- Model Evaluation & Visualization ----------
st.header("Model Evaluation & Visualization")
if st.checkbox("Show Confusion Matrix & ROC Curve (ML Model - Full Test Set)"):
    df = pd.read_csv("data/combined_news.csv")
    X_raw = df["combined_text"]
    y_true = df["label"]

    X_features = vectorizer.transform(X_raw)
    y_pred = ml_model.predict(X_features)
    if hasattr(ml_model, "predict_proba"):
        y_pred_probs = ml_model.predict_proba(X_features)[:, 1]
    else:
        y_pred_probs = None

    cm = confusion_matrix(y_true, y_pred, labels=["FAKE", "REAL"])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    if y_pred_probs is not None:
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true).ravel()
        fpr, tpr, _ = roc_curve(y_true_bin, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

st.markdown("""
*Tip: BERT evaluation and batch visualizations are best done in Python scripts (`bert_finetune.py`) to avoid performance issues in Streamlit. 
For live demo, use the prediction buttons above!*
""")

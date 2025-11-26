import pandas as pd
import re
import nltk
import spacy

nltk.download('stopwords')
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # remove special chars
    doc = nlp(text.lower())
    lemmatized = " ".join([token.lemma_ for token in doc if token.text not in stopwords.words('english')])
    return lemmatized


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)
    df['clean_text'] = df['combined_text'].apply(clean_text)
    return df

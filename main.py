import pandas as pd
from data_load_clean import load_and_clean_data  # if you use it
from feature_engineering import get_tfidf_features  # optional
from train_models import train_models, tune_model
from evaluation import evaluate_model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load your raw data as usual
df = pd.read_csv('data/combined_news.csv')
X_raw = df['combined_text']
y = df['label']

# 2. Convert text to numeric features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_raw)  # X is now numeric features for ML models

# 3. Proceed with train/test and modeling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train your models using train_models(X_train, y_train)
models, scores, splits = train_models(X_train, y_train)
best_name = max(scores, key=scores.get)
best_model = models[best_name]
joblib.dump(best_model, "best_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")  # <-- Save the trained vectorizer here!
print(f"Saved best model: {best_name}")

# 5. Evaluate model
X_train, X_test, y_train, y_test = splits
evaluate_model(best_model, X_test, y_test)
print("Ready to run the Streamlit UI or batch prediction.")

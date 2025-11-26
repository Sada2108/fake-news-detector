import pandas as pd


def batch_predict(csv_path, model, vectorizer, output_path="results.csv"):
    df = pd.read_csv(csv_path)
    df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)
    features = vectorizer.transform(df['combined_text'])
    df['prediction'] = model.predict(features)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

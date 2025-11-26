import pandas as pd

file_list = [
    'data/A-Table 1.csv',
    'data/IFND.csv',
    'data/news_dataset.csv',
    'data/news.csv',
    'data/Sheet1-Table 1.csv',
    'data/Sheet3-Table 1.csv'
]

dfs = []

for file in file_list:
    try:
        df = pd.read_csv(file)
        
        print(f"Columns in {file}: {list(df.columns)}")
        
        title_col = None
        for col in df.columns:
            if col.lower() in ['title', 'headline', 'news']:
                title_col = col
        text_col = None
        for col in df.columns:
            if col.lower() in ['text', 'content', 'body']:
                text_col = col
        label_col = None
        for col in df.columns:
            if col.lower() in ['label', 'class', 'target']:
                label_col = col
        if title_col and text_col and label_col:
            mini = df[[title_col, text_col, label_col]].copy()
            mini.columns = ['title', 'text', 'label']  
            dfs.append(mini)
        else:
            print(f"Skipping {file}. Not enough columns found.")
    except Exception as e:
        print(f"Failed to process {file}: {e}")


if len(dfs) == 0:
    print("No datasets found with required columns.")
else:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.dropna(subset=['title', 'text', 'label'], inplace=True)
    combined_df['combined_text'] = combined_df['title'].astype(str) + " " + combined_df['text'].astype(str)
    combined_df.to_csv('data/combined_news.csv', index=False)
    print(f"Saved merged dataset as data/combined_news.csv with {len(combined_df)} rows.")


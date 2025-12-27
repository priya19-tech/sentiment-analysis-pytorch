
import pandas as pd
from preprocess import clean_text

def load_imdb_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['review', 'sentiment']]
    
    # Clean text
    df['review'] = df['review'].apply(clean_text)
    
    # Encode labels
    df['sentiment'] = df['sentiment'].map({
        'positive': 1,
        'negative': 0
    })
    
    return df


if __name__ == "__main__":
    data = load_imdb_data("data/IMDB Dataset.csv")
    print(data.head())
    print(data['sentiment'].value_counts())

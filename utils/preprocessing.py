import pandas as pd

def load_and_clean_data(path="../data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"):

    df = pd.read_csv(path)
    df = df[['reviews.text', 'reviews.rating']].dropna()
    df.rename(columns={'reviews.text': 'text', 'reviews.rating': 'rating'}, inplace=True)
    df = df[df['rating'].isin([1, 2, 3, 4, 5])]

    def map_label(r):
        if r in [1, 2]:
            return 0  # Negative
        elif r == 3:
            return 1  # Neutral
        else:
            return 2  # Positive

    df['label'] = df['rating'].apply(map_label)
    return df[['text', 'label']]

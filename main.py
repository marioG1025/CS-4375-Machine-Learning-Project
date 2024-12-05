import pandas as pd

# Load and Clean Steam Data
steam_data = pd.read_csv("Datasets/steam.csv")
steam_data = steam_data.dropna(subset=['review_text'])  # Drop rows with missing reviews
steam_data = steam_data[['review_text', 'review_score']]  # Keep only relevant columns
steam_data = steam_data.rename(columns={'review_text': 'review', 'review_score': 'sentiment'})
steam_data['sentiment'] = steam_data['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')

# Load and Clean Yelp Data
yelp_data = pd.read_csv("Datasets/yelp.csv", header=None)  # Load Yelp Data without a header
yelp_data = yelp_data.iloc[:, :2]  # Select the first two columns
yelp_data.columns = ['review', 'sentiment']  # Rename columns
yelp_data['sentiment'] = pd.to_numeric(yelp_data['sentiment'], errors='coerce')  # Convert sentiment to numeric
valid_sentiments = {0: 'negative', 1: 'positive'}  # Map sentiment values
yelp_data['sentiment'] = yelp_data['sentiment'].map(valid_sentiments)
yelp_data = yelp_data.dropna(subset=['review', 'sentiment'])  # Drop rows with missing or invalid values

# Load and Clean IMDb Data
imdb_data = pd.read_csv("Datasets/imdb.csv")
imdb_data = imdb_data[['review', 'sentiment']]  # Keep only relevant columns

# Load and Clean Amazon Data
amazon_data = pd.read_csv("Datasets/amazon.csv")
amazon_data = amazon_data[['Text', 'Score']]  # Keep only relevant columns
amazon_data = amazon_data.rename(columns={'Text': 'review'})  # Rename columns
amazon_data['sentiment'] = amazon_data['Score'].apply(lambda x: 'positive' if x > 3 else 'negative')
amazon_data = amazon_data.drop(columns=['Score'])  # Drop Score after processing

# Save Cleaned Datasets
steam_data.to_csv("Datasets/cleaned_steam.csv", index=False)
yelp_data.to_csv("Datasets/cleaned_yelp.csv", index=False)
imdb_data.to_csv("Datasets/cleaned_imdb.csv", index=False)
amazon_data.to_csv("Datasets/cleaned_amazon.csv", index=False)

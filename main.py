
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
'''
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
###############
#####################
######
###############
#####################
######

# Tokenization Function with Progress Logging
def tokenize_reviews_nltk(df, review_column):
    total_rows = len(df)
    print(f"Starting tokenization for {total_rows} rows with NLTK...")
    for idx in range(total_rows):
        if idx % 1000 == 0 or idx == total_rows - 1:  # Progress log every 1000 rows or on the last row
            print(f"Tokenizing row {idx + 1} of {total_rows}...")
        df.loc[idx, review_column] = word_tokenize(str(df.loc[idx, review_column]).lower())
    return df

# Tokenize Cleaned Steam Data
print("Loading cleaned Steam data...")
steam_data = pd.read_csv("Datasets/cleaned_steam.csv")
print("Tokenizing Steam data with NLTK...")
steam_data = tokenize_reviews_nltk(steam_data, 'review')
print("Steam data tokenization complete!")

# Tokenize Cleaned Yelp Data
print("Loading cleaned Yelp data...")
yelp_data = pd.read_csv("Datasets/cleaned_yelp.csv")
print("Tokenizing Yelp data with NLTK...")
yelp_data = tokenize_reviews_nltk(yelp_data, 'review')
print("Yelp data tokenization complete!")

# Tokenize Cleaned IMDb Data
print("Loading cleaned IMDb data...")
imdb_data = pd.read_csv("Datasets/cleaned_imdb.csv")
print("Tokenizing IMDb data with NLTK...")
imdb_data = tokenize_reviews_nltk(imdb_data, 'review')
print("IMDb data tokenization complete!")

# Tokenize Cleaned Amazon Data
print("Loading cleaned Amazon data...")
amazon_data = pd.read_csv("Datasets/cleaned_amazon.csv")
print("Tokenizing Amazon data with NLTK...")
amazon_data = tokenize_reviews_nltk(amazon_data, 'review')
print("Amazon data tokenization complete!")

# Save Tokenized Data to CSV
print("Saving tokenized data...")
steam_data.to_csv("Datasets/tokenized_steam.csv", index=False)
yelp_data.to_csv("Datasets/tokenized_yelp.csv", index=False)
imdb_data.to_csv("Datasets/tokenized_imdb.csv", index=False)
amazon_data.to_csv("Datasets/tokenized_amazon.csv", index=False)
print("All tokenized data saved!")
'''
#####################
####################
##TF-ID##
################
############

# Load tokenized datasets
print("Loading tokenized Steam data...")
steam_data = pd.read_csv("Datasets/tokenized_steam.csv")
print("Loading tokenized Yelp data...")
yelp_data = pd.read_csv("Datasets/tokenized_yelp.csv")
print("Loading tokenized IMDb data...")
imdb_data = pd.read_csv("Datasets/tokenized_imdb.csv")
print("Loading tokenized Amazon data...")
amazon_data = pd.read_csv("Datasets/tokenized_amazon.csv")

print("Converting tokenized reviews to strings...")
steam_data['review'] = steam_data['review'].apply(lambda tokens: ' '.join(eval(tokens)))
yelp_data['review'] = yelp_data['review'].apply(lambda tokens: ' '.join(eval(tokens)))
imdb_data['review'] = imdb_data['review'].apply(lambda tokens: ' '.join(eval(tokens)))
amazon_data['review'] = amazon_data['review'].apply(lambda tokens: ' '.join(eval(tokens)))

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_reviews(data, review_column, max_features=5000):
    print(f"Applying TF-IDF vectorization (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(data[review_column])
    print("TF-IDF vectorization complete!")
    return tfidf_matrix, vectorizer

# Vectorize each dataset
print("Vectorizing Steam data...")
steam_tfidf, steam_vectorizer = vectorize_reviews(steam_data, 'review')

print("Vectorizing Yelp data...")
yelp_tfidf, yelp_vectorizer = vectorize_reviews(yelp_data, 'review')

print("Vectorizing IMDb data...")
imdb_tfidf, imdb_vectorizer = vectorize_reviews(imdb_data, 'review')

print("Vectorizing Amazon data...")
amazon_tfidf, amazon_vectorizer = vectorize_reviews(amazon_data, 'review')

import joblib

# Save TF-IDF matrices and vectorizers
print("Saving TF-IDF matrices and vectorizers...")
joblib.dump(steam_tfidf, "Datasets/steam_tfidf.pkl")
joblib.dump(steam_vectorizer, "Datasets/steam_vectorizer.pkl")
joblib.dump(yelp_tfidf, "Datasets/yelp_tfidf.pkl")
joblib.dump(yelp_vectorizer, "Datasets/yelp_vectorizer.pkl")
joblib.dump(imdb_tfidf, "Datasets/imdb_tfidf.pkl")
joblib.dump(imdb_vectorizer, "Datasets/imdb_vectorizer.pkl")
joblib.dump(amazon_tfidf, "Datasets/amazon_tfidf.pkl")
joblib.dump(amazon_vectorizer, "Datasets/amazon_vectorizer.pkl")
print("TF-IDF data saved!")

#####################
####################
##DATA SPLIT##
################
############

from sklearn.model_selection import train_test_split
import joblib

# Load TF-IDF matrix and sentiment labels
print("Loading TF-IDF matrix and labels for Steam data...")
steam_tfidf = joblib.load("Datasets/steam_tfidf.pkl")
steam_data = pd.read_csv("Datasets/tokenized_steam.csv")
steam_labels = steam_data['sentiment'].map({'positive': 1, 'negative': 0})  # Map labels to binary values

# Split Steam Data
print("Splitting Steam data into training and testing sets...")
X_train_steam, X_test_steam, y_train_steam, y_test_steam = train_test_split(
    steam_tfidf, steam_labels, test_size=0.2, random_state=42
)
print(f"Steam data split: {X_train_steam.shape[0]} training rows, {X_test_steam.shape[0]} testing rows.")

# Repeat for Yelp Data
print("Loading TF-IDF matrix and labels for Yelp data...")
yelp_tfidf = joblib.load("Datasets/yelp_tfidf.pkl")
yelp_data = pd.read_csv("Datasets/tokenized_yelp.csv")
yelp_labels = yelp_data['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting Yelp data into training and testing sets...")
X_train_yelp, X_test_yelp, y_train_yelp, y_test_yelp = train_test_split(
    yelp_tfidf, yelp_labels, test_size=0.2, random_state=42
)
print(f"Yelp data split: {X_train_yelp.shape[0]} training rows, {X_test_yelp.shape[0]} testing rows.")

# Repeat for IMDb Data
print("Loading TF-IDF matrix and labels for IMDb data...")
imdb_tfidf = joblib.load("Datasets/imdb_tfidf.pkl")
imdb_data = pd.read_csv("Datasets/tokenized_imdb.csv")
imdb_labels = imdb_data['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting IMDb data into training and testing sets...")
X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(
    imdb_tfidf, imdb_labels, test_size=0.2, random_state=42
)
print(f"IMDb data split: {X_train_imdb.shape[0]} training rows, {X_test_imdb.shape[0]} testing rows.")

# Repeat for Amazon Data
print("Loading TF-IDF matrix and labels for Amazon data...")
amazon_tfidf = joblib.load("Datasets/amazon_tfidf.pkl")
amazon_data = pd.read_csv("Datasets/tokenized_amazon.csv")
amazon_labels = amazon_data['sentiment'].map({'positive': 1, 'negative': 0})

print("Splitting Amazon data into training and testing sets...")
X_train_amazon, X_test_amazon, y_train_amazon, y_test_amazon = train_test_split(
    amazon_tfidf, amazon_labels, test_size=0.2, random_state=42
)
print(f"Amazon data split: {X_train_amazon.shape[0]} training rows, {X_test_amazon.shape[0]} testing rows.")

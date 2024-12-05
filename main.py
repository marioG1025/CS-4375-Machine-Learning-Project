import pandas as pd

# Load datasets
steam_data = pd.read_csv("Datasets/steam.csv")
yelp_data = pd.read_csv("Datasets/yelp.csv")
imdb_data = pd.read_csv("Datasets/imdb.csv")
amazon_data = pd.read_csv("Datasets/amazon.csv")

# Preview the datasets
print("Steam Data:")
print(steam_data.head(), "\n")

print("Yelp Data:")
print(yelp_data.head(), "\n")

print("IMDB Data:")
print(imdb_data.head(), "\n")

print("Amazon Data:")
print(amazon_data.head(), "\n")

print(steam_data.isnull().sum())
print(yelp_data.isnull().sum())
print(imdb_data.isnull().sum())
print(amazon_data.isnull().sum())


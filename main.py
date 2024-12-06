
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
'''
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



#############
############
##MODEL TRAINING
#################
####################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train and Evaluate a Model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, dataset_name):
    print(f"Training Logistic Regression model for {dataset_name}...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Model Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")

    print(f"Classification Report for {dataset_name}:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix for {dataset_name}:\n", confusion_matrix(y_test, y_pred))

    return model

# Train and Evaluate on Steam Data
print("Processing Steam data...")
steam_model = train_and_evaluate_model(X_train_steam, X_test_steam, y_train_steam, y_test_steam, "Steam")

# Train and Evaluate on Yelp Data
print("Processing Yelp data...")
yelp_model = train_and_evaluate_model(X_train_yelp, X_test_yelp, y_train_yelp, y_test_yelp, "Yelp")

# Train and Evaluate on IMDb Data
print("Processing IMDb data...")
imdb_model = train_and_evaluate_model(X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb, "IMDb")

# Train and Evaluate on Amazon Data
print("Processing Amazon data...")
amazon_model = train_and_evaluate_model(X_train_amazon, X_test_amazon, y_train_amazon, y_test_amazon, "Amazon")

from sklearn.naive_bayes import MultinomialNB

# Train and Evaluate a Naive Bayes Model
def train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test, dataset_name):
    print(f"Training Naive Bayes model for {dataset_name}...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Model Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} Naive Bayes Accuracy: {accuracy:.4f}")

    print(f"Classification Report for {dataset_name}:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix for {dataset_name}:\n", confusion_matrix(y_test, y_pred))

    return model

# Train and Evaluate on Steam Data
print("Processing Steam data with Naive Bayes...")
steam_nb_model = train_and_evaluate_naive_bayes(X_train_steam, X_test_steam, y_train_steam, y_test_steam, "Steam")

# Train and Evaluate on Yelp Data
print("Processing Yelp data with Naive Bayes...")
yelp_nb_model = train_and_evaluate_naive_bayes(X_train_yelp, X_test_yelp, y_train_yelp, y_test_yelp, "Yelp")

# Train and Evaluate on IMDb Data
print("Processing IMDb data with Naive Bayes...")
imdb_nb_model = train_and_evaluate_naive_bayes(X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb, "IMDb")

# Train and Evaluate on Amazon Data
print("Processing Amazon data with Naive Bayes...")
amazon_nb_model = train_and_evaluate_naive_bayes(X_train_amazon, X_test_amazon, y_train_amazon, y_test_amazon, "Amazon")

from sklearn.svm import LinearSVC

# Train and Evaluate an SVM Model
def train_and_evaluate_svm(X_train, X_test, y_train, y_test, dataset_name):
    print(f"Training SVM model for {dataset_name}...")
    model = LinearSVC(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # Model Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{dataset_name} SVM Accuracy: {accuracy:.4f}")

    print(f"Classification Report for {dataset_name}:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix for {dataset_name}:\n", confusion_matrix(y_test, y_pred))

    return model

# Train and Evaluate on Steam Data
print("Processing Steam data with SVM...")
steam_svm_model = train_and_evaluate_svm(X_train_steam, X_test_steam, y_train_steam, y_test_steam, "Steam")

# Train and Evaluate on Yelp Data
print("Processing Yelp data with SVM...")
yelp_svm_model = train_and_evaluate_svm(X_train_yelp, X_test_yelp, y_train_yelp, y_test_yelp, "Yelp")

# Train and Evaluate on IMDb Data
print("Processing IMDb data with SVM...")
imdb_svm_model = train_and_evaluate_svm(X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb, "IMDb")

# Train and Evaluate on Amazon Data
print("Processing Amazon data with SVM...")
amazon_svm_model = train_and_evaluate_svm(X_train_amazon, X_test_amazon, y_train_amazon, y_test_amazon, "Amazon")


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Function to evaluate and store metrics
def evaluate_model(model, X_test, y_test, dataset_name, model_name):
    print(f"Evaluating {model_name} on {dataset_name} data...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)  # Get as dictionary

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision (0)": report['0']['precision'],
        "Recall (0)": report['0']['recall'],
        "F1-Score (0)": report['0']['f1-score'],
        "Precision (1)": report['1']['precision'],
        "Recall (1)": report['1']['recall'],
        "F1-Score (1)": report['1']['f1-score']
    }

# Collect results for all models and datasets
results = []

# Evaluate Steam Models
results.append(evaluate_model(steam_model, X_test_steam, y_test_steam, "Steam", "Logistic Regression"))
results.append(evaluate_model(steam_nb_model, X_test_steam, y_test_steam, "Steam", "Naive Bayes"))
results.append(evaluate_model(steam_svm_model, X_test_steam, y_test_steam, "Steam", "SVM"))

# Evaluate Yelp Models
results.append(evaluate_model(yelp_model, X_test_yelp, y_test_yelp, "Yelp", "Logistic Regression"))
results.append(evaluate_model(yelp_nb_model, X_test_yelp, y_test_yelp, "Yelp", "Naive Bayes"))
results.append(evaluate_model(yelp_svm_model, X_test_yelp, y_test_yelp, "Yelp", "SVM"))

# Evaluate IMDb Models
results.append(evaluate_model(imdb_model, X_test_imdb, y_test_imdb, "IMDb", "Logistic Regression"))
results.append(evaluate_model(imdb_nb_model, X_test_imdb, y_test_imdb, "IMDb", "Naive Bayes"))
results.append(evaluate_model(imdb_svm_model, X_test_imdb, y_test_imdb, "IMDb", "SVM"))

# Evaluate Amazon Models
results.append(evaluate_model(amazon_model, X_test_amazon, y_test_amazon, "Amazon", "Logistic Regression"))
results.append(evaluate_model(amazon_nb_model, X_test_amazon, y_test_amazon, "Amazon", "Naive Bayes"))
results.append(evaluate_model(amazon_svm_model, X_test_amazon, y_test_amazon, "Amazon", "SVM"))

# Create a summary table
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df)

# Save results to CSV for future reference
results_df.to_csv("Model_Performance_Summary.csv", index=False)

#The F1 score is a performance metric that combines precision and recall into a single value.
# $It is particularly useful when you want a balance between the two
#1 Score ranges from 0 to 1:
#1: Perfect balance between precision and recall.
#0: No true positives at all.
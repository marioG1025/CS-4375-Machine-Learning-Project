# Sentiment Analysis and Text Classification Project

## Overview
This project focuses on building a text classification model to analyze sentiment from text data.<br>
We use datasets from multiple domains, including movies, e-commerce, restaurant reviews, and video games, to explore sentiment trends and evaluate classification techniques.

## Objectives
- Preprocess text data (e.g., tokenization, stop-word removal, lemmatization).
- Experiment with feature extraction techniques like Bag of Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF).
- Train and compare machine learning classifiers (e.g., Naive Bayes, Logistic Regression, Random Forest).
- Evaluate model performance across different datasets.

---

## Datasets Used
1. **IMDb Dataset of 50K Movie Reviews**  
   - Sentiment analysis on movie reviews.
   - Labels: Positive, Negative.

2. **Amazon Reviews Dataset**  
   - Sentiment analysis on product reviews.
   - Labels: Positive, Negative.

3. **Yelp Reviews Sentiment Dataset**  
   - Sentiment analysis on restaurant reviews.
   - Labels: Positive, Negative.

4. **Sentiment Analysis for Steam Reviews**  
   - Sentiment analysis on video game reviews.
   - Labels: Positive, Negative.

---
## Installation and Setup
1. **Clone the repository**
   - `git clone https://github.com/marioG1025/CS-4375-Machine-Learning-Project.git `

2. **Install required Python packages:**
   - `pip install pandas numpy scikit-learn nltk matplotlib seaborn`
   - `pip install kaggle`
   - `pip install opendatasets --upgrade --quiet`


## Using the Kaggle API to Download Datasets

**Follow these steps to set up the Kaggle API and download the required datasets for this project.**

1. Install the Kaggle API
   Make sure you have Python installed. Then, install the Kaggle API by running:
   - `pip install kaggle`
     
3. Get Your Kaggle API Key
   - Log in to your Kaggle account at kaggle.com.
   - Navigate to your account settings:
   - Click on your profile picture in the top-right corner and select Account.
   - Scroll down to the API section and click on Create New API Token.
   - This will download a file named kaggle.json to your computer.
     
4. Configure the Kaggle API
   - Move the kaggle.json file to the correct location:
   - **For Unix/macOS:**
   - mkdir -p ~/.kaggle
   - mv ~/Downloads/kaggle.json ~/.kaggle/
   - chmod 600 ~/.kaggle/kaggle.json
     
   **For Windows:**
   - mkdir %USERPROFILE%\.kaggle
   - move %HOMEPATH%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
   - Ensure the API key file is in place before running Kaggle commands.
  
5. Download Datasets
   - Use the Kaggle API to download datasets. For each dataset, use the specific Kaggle dataset identifier.<br> 

Below are examples for this project:
1. IMDb Dataset of 50K Movie Reviews:
`kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`
2. Amazon Reviews Dataset:
`kaggle datasets download -d snap/amazon-fine-food-reviews`
3. Yelp Reviews Sentiment Dataset:
`kaggle datasets download -d marklvl/sentiment-labelled-sentences-data-set`
4. Sentiment Analysis for Steam Reviews:
`kaggle datasets download -d andrewmvd/steam-reviews`

5. Extract the Downloaded Files
   - The datasets will be downloaded as .zip files. Extract them using the following command:
   - unzip dataset-file-name.zip<br>
   
For example:
unzip imdb-dataset-of-50k-movie-reviews.zip
6. Verify the Files
   After extraction, you should see the .csv files in your project directory. Use ls (or File Explorer on Windows) to check if the files are present.

7. Loading the Dataset in Python
   Once the .csv files are downloaded and extracted, you can load them into your Python script using pandas:

import pandas as pd

# Load the IMDb dataset
imdb_data = pd.read_csv("IMDB_Dataset.csv")
print(imdb_data.head())
By following these steps, you and your team can efficiently set up and download datasets for this project. Let me know if you'd like to refine this further!



## Key Technologies and Tools
- **Programming Language**: Python
- **Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `nltk` for text preprocessing.
  - `scikit-learn` for feature extraction and model training.
  - `matplotlib` for visualizations.
    `


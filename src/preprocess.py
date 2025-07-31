import pandas as pd
import re
import os
import argparse
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def clean_text(text):
    if pd.isna(text):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_csv(input_path, output_path, text_column):
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")
    df[text_column] = df[text_column].apply(lambda x: clean_text(x))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    preprocess_csv(
        input_path="./data/raw/unfair_tos_test.csv",
        output_path="./data/processed/unfair_tos_test_cleaned.csv",
        text_column="text",
    )
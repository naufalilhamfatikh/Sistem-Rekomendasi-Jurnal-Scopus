import dill
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fungsi untuk memuat model dengan aman
def load_model_safely(model_path):
    # Definisikan skeleton class dan fungsi untuk handle unpickling
    class SimilarityEstimator(BaseEstimator, TransformerMixin):
        def __init__(self, tfidf=None, svd=None):
            self.tfidf = tfidf
            self.svd = svd

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def score(self, X, y=None):
            return 0.0

    # Pastikan komponen terdaftar di sys.modules
    sys.modules['__main__'].SimilarityEstimator = SimilarityEstimator
    
    with open(model_path, 'rb') as f:
        model_data = dill.load(f)
    
    return model_data

# Fungsi preprocessing (salin dari kode asli)
def preprocess_text(text):
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download resource jika belum ada
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    
    # Proses teks
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    if pd.isna(text):
        return "placeholder"

    text = str(text)
    if not text.strip():
        return "placeholder"

    text = text.replace('/', ' ')
    text = re.sub(r'[()]', ' ', text)
    text = text.replace(';', ' ')
    text = text.replace('-', ' ')
    text = re.sub(r'[^\w\s,]', '', text)
    text = text.replace(',', ' ')
    text = re.sub(r'\d+', '', text)  # Remove numbers

    text_parts = re.split(r',\s*', text)
    processed_parts = []
    lemmatizer = WordNetLemmatizer()

    for part in text_parts:
        part = part.lower()
        tokens = word_tokenize(part)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_part = ' '.join(tokens)
        if processed_part:
            processed_parts.append(processed_part)

    return ' '.join(processed_parts) if processed_parts else "placeholder"

def get_recommendations_with_price_filter(model, df, query_abstract, top_n=15, min_price=None, max_price=None):
    # Dapatkan rekomendasi awal
    recommendations = get_recommendations(model, df, query_abstract, top_n)
    
    # Filter berdasarkan harga
    if min_price is not None:
        recommendations = recommendations[recommendations[' APC (Biaya Publikasi)'] >= min_price]
    if max_price is not None:
        recommendations = recommendations[recommendations[' APC (Biaya Publikasi)'] <= max_price]
    
    # Pastikan tetap terurut berdasarkan similarity
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    
    return recommendations.head(top_n)
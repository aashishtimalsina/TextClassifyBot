import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class DataProcessor:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        # Download necessary NLTK resources
        try:
            # Download punkt instead of punkt_tab
            nltk.download('punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        
    def load_data(self, filename):
        """
        Load data from a CSV file
        """
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Data file {filename} not found in {self.data_dir}")
    
    def preprocess_text(self, text):
        """
        Preprocess text data: lowercase, remove special chars, 
        remove stopwords, lemmatize
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize using a simpler string split method to avoid nltk punkt_tab dependency
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(word) 
                         for word in tokens 
                         if word not in self.stop_words]
        
        return ' '.join(cleaned_tokens)
    
    def prepare_data(self, df, text_column, label_column, test_size=0.2, random_state=42):
        """
        Prepare data for training: preprocess, vectorize, and split
        """
        # Preprocess text
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df[label_column], 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[label_column] if len(df[label_column].unique()) > 1 else None
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test
    
    def save_vectorizer(self, model_dir='../models'):
        """
        Save the vectorizer for later use in predictions
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    
    def load_vectorizer(self, model_dir='../models'):
        """
        Load a saved vectorizer
        """
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
    
    def create_sample_data(self, save=True):
        """
        Create a sample dataset for text classification
        """
        # Sample data: text messages with their category (spam/not spam)
        data = {
            'message': [
                "Free entry to win a prize! Text YES to 1234",
                "Call now for a great deal on insurance",
                "Meeting at 3pm tomorrow in the conference room",
                "URGENT: Your account has been compromised",
                "Hi John, can we reschedule our meeting?",
                "Congratulations! You've been selected for a free gift",
                "Please bring the reports for tomorrow's presentation",
                "FINAL NOTICE: Your payment is due",
                "Don't forget to pick up milk on your way home",
                "Amazing opportunity to earn money from home"
            ],
            'category': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 for spam, 0 for not spam
        }
        
        df = pd.DataFrame(data)
        
        if save:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            df.to_csv(os.path.join(self.data_dir, 'sample_spam_data.csv'), index=False)
            
        return df
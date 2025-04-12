import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import joblib

from data_processor import DataProcessor
from model import CustomTextClassifier, SimpleTextClassifierNN

class ModelTrainer:
    def __init__(self, data_dir='../data', models_dir='../models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.data_processor = DataProcessor(data_dir=data_dir)
        self.model = None
        self.training_history = None
        self.metrics = {}
        
    def train(self, model_type='custom', data_file=None, text_column='message', 
              label_column='category', epochs=10, batch_size=32, hidden_layers=[64, 32]):
        """
        Train a model on the given data
        """
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Load data or create sample data if not provided
        if data_file is None:
            print("No data file provided. Creating sample data...")
            df = self.data_processor.create_sample_data(save=True)
            data_file = 'sample_spam_data.csv'
        else:
            df = self.data_processor.load_data(data_file)
        
        print(f"Data loaded: {df.shape[0]} samples")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data(
            df, text_column, label_column
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Initialize model
        if model_type.lower() == 'custom':
            self.model = CustomTextClassifier(
                epochs=epochs, 
                batch_size=batch_size, 
                hidden_layers=hidden_layers
            )
            
            # Train model
            start_time = time.time()
            history = self.model.fit(X_train, y_train, validation_split=0.1)
            self.training_history = history
            training_time = time.time() - start_time
            
            print(f"Model trained in {training_time:.2f} seconds")
            
            # Save vectorizer for future use
            self.data_processor.save_vectorizer(model_dir=self.models_dir)
            
            # Save model
            self.model.save(model_dir=self.models_dir)
            
        elif model_type.lower() == 'simple':
            # Convert sparse matrices to dense arrays for the simple model
            if hasattr(X_train, 'toarray'):
                X_train = X_train.toarray()
            if hasattr(X_test, 'toarray'):
                X_test = X_test.toarray()
            
            # Initialize simple model with input dimensions from data
            self.model = SimpleTextClassifierNN(
                input_dim=X_train.shape[1], 
                hidden_dim=32, 
                output_dim=1
            )
            
            # Train model
            start_time = time.time()
            losses = self.model.train(X_train, y_train.values.reshape(-1, 1), 
                                    learning_rate=0.01, epochs=epochs)
            self.training_history = losses
            training_time = time.time() - start_time
            
            print(f"Model trained in {training_time:.2f} seconds")
            
            # Save vectorizer for future use
            self.data_processor.save_vectorizer(model_dir=self.models_dir)
            
            # Save model
            self.model.save(model_dir=self.models_dir)
            
            # Plot training loss
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(self.models_dir, 'training_loss.png'))
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate model
        self.evaluate(X_test, y_test)
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        # Predict on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Print metrics
        print("\nModel Evaluation:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        # Save metrics
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        joblib.dump(self.metrics, os.path.join(self.models_dir, 'metrics.pkl'))
        
        return self.metrics
    
    def load_model(self, model_type='custom'):
        """
        Load a trained model
        """
        if model_type.lower() == 'custom':
            self.model = CustomTextClassifier()
            self.model.load(model_dir=self.models_dir)
        elif model_type.lower() == 'simple':
            self.model = SimpleTextClassifierNN()
            self.model.load(model_dir=self.models_dir)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load vectorizer
        self.data_processor.load_vectorizer(model_dir=self.models_dir)
        
        return self.model
    
    def predict(self, texts, model_type='custom'):
        """
        Make predictions on new texts
        """
        if self.model is None:
            self.load_model(model_type=model_type)
        
        # Preprocess and vectorize texts
        processed_texts = [self.data_processor.preprocess_text(text) for text in texts]
        X = self.data_processor.vectorizer.transform(processed_texts)
        
        # Predict
        y_pred = self.model.predict(X)
        
        return y_pred
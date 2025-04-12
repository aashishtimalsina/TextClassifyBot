import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

class CustomTextClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom text classifier that combines TF-IDF features with neural network
    """
    def __init__(self, input_dim=1000, hidden_layers=[64, 32], dropout_rate=0.3, 
                 learning_rate=0.001, epochs=10, batch_size=32):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        
    def _build_model(self, input_shape):
        """
        Build neural network model architecture
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], input_shape=(input_shape,), 
                        activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    
    def fit(self, X, y, validation_split=0.1, verbose=1):
        """
        Fit the model to the training data
        """
        # Get input shape from data
        input_shape = X.shape[1]
        self.input_dim = input_shape
        
        # Build model
        self.model = self._build_model(input_shape)
        
        # Convert sparse matrix to dense if necessary
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """
        Predict classes for samples in X
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Predict probabilities
        y_prob = self.model.predict(X)
        
        # Convert probabilities to class labels
        y_pred = (y_prob > 0.5).astype(int).flatten()
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Predict probabilities for class 1
        y_prob1 = self.model.predict(X).flatten()
        
        # Stack probabilities for both classes
        return np.vstack((1 - y_prob1, y_prob1)).T
    
    def save(self, model_dir='../models'):
        """
        Save the trained model
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model.save(os.path.join(model_dir, 'text_classifier.h5'))
        
        # Save other parameters
        params = {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        
        joblib.dump(params, os.path.join(model_dir, 'classifier_params.pkl'))
    
    def load(self, model_dir='../models'):
        """
        Load a trained model
        """
        # Load model parameters
        params_path = os.path.join(model_dir, 'classifier_params.pkl')
        if os.path.exists(params_path):
            params = joblib.load(params_path)
            self.input_dim = params['input_dim']
            self.hidden_layers = params['hidden_layers']
            self.dropout_rate = params['dropout_rate']
            self.learning_rate = params['learning_rate']
            self.epochs = params['epochs']
            self.batch_size = params['batch_size']
        
        # Load model
        model_path = os.path.join(model_dir, 'text_classifier.h5')
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")


class SimpleTextClassifierNN:
    """
    A simpler text classifier with neural network for easier testing
    """
    def __init__(self, input_dim=100, hidden_dim=32, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights randomly
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def forward(self, X):
        """Forward pass through the network"""
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        m = y_true.shape[0]
        loss = -1/m * np.sum(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))
        return loss
    
    def backward(self, X, y):
        """Backward pass to compute gradients"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = 1/m * np.dot(self.a1.T, dz2)
        db2 = 1/m * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)  # ReLU derivative
        dW1 = 1/m * np.dot(X.T, dz1)
        db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, learning_rate=0.01, epochs=1000, verbose=True):
        """Train the model"""
        losses = []
        
        for i in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y)
            
            # Update parameters
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            # Print progress
            if verbose and i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)
        
    def save(self, model_dir='../models'):
        """Save the model parameters"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim
        }
        
        joblib.dump(params, os.path.join(model_dir, 'simple_nn_params.pkl'))
    
    def load(self, model_dir='../models'):
        """Load model parameters"""
        params_path = os.path.join(model_dir, 'simple_nn_params.pkl')
        if os.path.exists(params_path):
            params = joblib.load(params_path)
            self.W1 = params['W1']
            self.b1 = params['b1']
            self.W2 = params['W2']
            self.b2 = params['b2']
            self.input_dim = params['input_dim']
            self.hidden_dim = params['hidden_dim']
            self.output_dim = params['output_dim']
        else:
            raise FileNotFoundError(f"Model parameters not found at {params_path}")
import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import from the src directory
from src.automator import Automator
from src.trainer import ModelTrainer

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize the automator
automator = Automator(data_dir='../data', models_dir='../models')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train a new model with parameters from the form"""
    try:
        # Get parameters from the form
        model_type = request.form.get('model_type', 'custom')
        epochs = int(request.form.get('epochs', 5))
        batch_size = int(request.form.get('batch_size', 32))
        hidden_layers = request.form.get('hidden_layers', '[64, 32]')
        
        # Parse hidden_layers from string to list
        try:
            hidden_layers = json.loads(hidden_layers)
        except:
            hidden_layers = [64, 32]
        
        # Train the model
        model, metrics = automator.train_model(
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            hyperparams={'hidden_layers': hidden_layers}
        )
        
        # Return the metrics
        if metrics:
            return jsonify({
                'success': True,
                'message': 'Model trained successfully!',
                'metrics': metrics
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed. Check logs for details.'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during model training: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction on input text"""
    try:
        # Get text from the form
        text = request.form.get('text', '')
        model_type = request.form.get('model_type', 'custom')
        
        if not text:
            return jsonify({
                'success': False,
                'message': 'Please provide text for prediction.'
            })
        
        # Make prediction
        predictions = automator.predict_batch([text], model_type=model_type)
        
        if predictions is not None:
            result = 'SPAM' if predictions[0] == 1 else 'NOT SPAM'
            
            return jsonify({
                'success': True,
                'message': 'Prediction made successfully!',
                'result': result,
                'prediction': int(predictions[0])
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Prediction failed. Check logs for details.'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during prediction: {str(e)}'
        })

@app.route('/get_metrics')
def get_metrics():
    """Get the latest model metrics"""
    try:
        # Load experiment history
        history = automator.load_experiment_history()
        
        if history and len(history) > 0:
            # Get the latest experiment
            latest = history[-1]
            
            # Return metrics
            return jsonify({
                'success': True,
                'metrics': latest.get('metrics', {}),
                'timestamp': latest.get('timestamp', ''),
                'model_type': latest.get('model_type', ''),
                'epochs': latest.get('epochs', 0),
                'batch_size': latest.get('batch_size', 0)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No experiment history found.'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving metrics: {str(e)}'
        })

@app.route('/get_data_sample')
def get_data_sample():
    """Get a sample of the training data"""
    try:
        data_path = os.path.join('..', 'data', 'sample_spam_data.csv')
        
        if os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
            
            # Convert to dictionary
            data = {
                'messages': df['message'].tolist(),
                'labels': df['category'].tolist()
            }
            
            return jsonify({
                'success': True,
                'data': data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Sample data file not found.'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving sample data: {str(e)}'
        })

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    """Run an auto-experiment with specified parameters"""
    try:
        # Get parameters
        model_types = request.form.getlist('model_types[]') or ['custom', 'simple']
        epoch_options = request.form.getlist('epoch_options[]') 
        batch_sizes = request.form.getlist('batch_sizes[]')
        
        # Convert string lists to integers
        if epoch_options:
            epoch_options = [int(e) for e in epoch_options]
        else:
            epoch_options = [5, 10]
            
        if batch_sizes:
            batch_sizes = [int(b) for b in batch_sizes]
        else:
            batch_sizes = [16, 32]
        
        # Start the experiment (don't wait for completion)
        # In a production environment, this would be better handled with a background task
        best_model = automator.auto_experiment(
            model_types=model_types,
            epoch_options=epoch_options,
            batch_sizes=batch_sizes
        )
        
        if best_model:
            return jsonify({
                'success': True,
                'message': 'Auto-experiment completed successfully!',
                'best_model': best_model
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Auto-experiment did not find a successful model. Check logs for details.'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during auto-experiment: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
# Mini AI - Text Classification Project

## Overview
Mini AI is a text classification project designed to classify text messages as SPAM or NOT SPAM using machine learning models. The project includes a backend for training and predicting with models, as well as a web-based frontend for user interaction.

## Features
- Train custom neural network models for text classification.
- Predict whether a message is SPAM or NOT SPAM.
- Visualize model metrics and experiment results.
- Auto-experimentation to find the best model configuration.
- Web-based user interface built with Flask.

## Project Structure
```
Mini_AI/
├── automator.log                # Log file for automation processes
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── data/                        # Dataset folder
│   └── sample_spam_data.csv     # Sample dataset for training
├── models/                      # Trained models and metrics
│   ├── classifier_params.pkl    # Model parameters
│   ├── experiment_history.csv   # Experiment history (CSV)
│   ├── experiment_history.pkl   # Experiment history (Pickle)
│   ├── metrics.pkl              # Model metrics
│   ├── text_classifier.h5       # Trained model file
│   └── vectorizer.pkl           # Text vectorizer
├── src/                         # Source code for backend
│   ├── __init__.py              # Package initializer
│   ├── automator.py             # Automation script
│   ├── data_processor.py        # Data processing utilities
│   ├── model.py                 # Model definitions
│   ├── trainer.py               # Model training utilities
│   └── __pycache__/             # Compiled Python files
├── web/                         # Web interface
│   ├── app.py                   # Flask application
│   ├── static/                  # Static files (CSS, JS, images)
│   │   ├── css/
│   │   │   └── style.css        # Stylesheet
│   │   ├── images/              # Placeholder for images
│   │   └── js/
│   │       └── script.js        # JavaScript for frontend
│   └── templates/               # HTML templates
│       └── index.html           # Main HTML template
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Mini_AI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Backend
1. Navigate to the `web` directory:
   ```bash
   cd web
   ```

2. Start the Flask application:
   ```bash
   python -m web.app
   ```

3. Open your browser and go to `http://127.0.0.1:5000` to access the web interface.

### Training a Model
- Use the "Train Model" tab in the web interface to configure and train a new model.

### Making Predictions
- Use the "Predict" tab in the web interface to classify text messages as SPAM or NOT SPAM.

### Viewing Metrics
- Use the "Metrics" tab to view the performance metrics of the latest trained model.

### Running Auto-Experiments
- Use the "Auto-Experiment" tab to run multiple experiments and find the best model configuration.

## Dependencies
- Python 3.8+
- Flask
- TensorFlow
- NLTK
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
# TextClassifyBot

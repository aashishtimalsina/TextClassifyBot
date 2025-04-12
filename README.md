# Mini AI Project: Text Classification with Automation

A Python-based mini AI project for text classification with custom neural network models and automated training capabilities. This project implements a spam detection system that can be used as a foundation for other text classification tasks.

## Features

- Custom neural network model implementation for text classification
- Text data processing pipeline with NLTK
- Automated model training and evaluation
- Experiment history tracking and visualization
- Scheduled training and model monitoring
- Simple and TensorFlow-based model implementations

## Project Structure

```
MINI_AI/
├── data/               # Data storage directory
├── models/             # Trained models and experiment history
├── src/                # Source code
│   ├── __init__.py
│   ├── automator.py    # Automation and scheduling
│   ├── data_processor.py # Data loading and preprocessing
│   ├── model.py        # Model implementations
│   └── trainer.py      # Model training utilities
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the automator script to train a model and make predictions:

```bash
cd src
python automator.py
```

This will:
1. Create a sample dataset (spam vs. non-spam messages)
2. Train a custom neural network model
3. Make predictions on example texts

### Custom Training

```python
from src.automator import Automator

# Initialize the automator
automator = Automator()

# Train with custom parameters
model, metrics = automator.train_model(
    model_type='custom',  # 'custom' or 'simple'
    epochs=20,
    batch_size=32,
    hyperparams={'hidden_layers': [128, 64, 32]}
)

# Make predictions
texts = [
    "Meeting scheduled for tomorrow at 10 AM",
    "URGENT: You've won a prize! Click here to claim"
]
predictions = automator.predict_batch(texts)
```

### Automated Experimentation

To find the best model configuration:

```python
# Run experiments with different parameters
best_model = automator.auto_experiment(
    model_types=['custom', 'simple'],
    epoch_options=[5, 10, 20],
    batch_sizes=[16, 32, 64]
)

# Plot experiment history
automator.plot_experiment_history(metric='accuracy')
```

### Scheduling Training Jobs

```python
# Schedule daily training
job_id = automator.schedule_training(
    interval='daily',
    time='02:00',
    epochs=10,
    batch_size=32
)

# Run the scheduler (blocking)
automator.run_scheduler()

# Or run non-blocking
automator.run_scheduler(blocking=False)
```

## Models

### CustomTextClassifier

A neural network model built with TensorFlow/Keras for text classification:
- Accepts TF-IDF features as input
- Configurable hidden layers and dropout
- Binary classification output

### SimpleTextClassifierNN

A simpler neural network implementation:
- Pure NumPy implementation (no TensorFlow required)
- Single hidden layer with ReLU activation
- Sigmoid output layer for binary classification

## Data Processing

The `DataProcessor` class handles:
- Text cleaning and normalization
- Stopword removal and lemmatization
- TF-IDF vectorization
- Train/test splitting

## Customization

The project can be extended for different text classification tasks by:

1. Providing custom datasets with text and label columns
2. Adjusting the preprocessing pipeline in `data_processor.py`
3. Modifying the model architecture in `model.py`
4. Using the automation tools in `automator.py` to find optimal configurations

## License

MIT# TextClassifyBot

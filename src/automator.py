import os
import time
import schedule
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import logging
from pathlib import Path

from data_processor import DataProcessor
from model import CustomTextClassifier
from trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../automator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MiniAI-Automator')

class Automator:
    """
    Automates the process of training and evaluating models
    """
    def __init__(self, data_dir='../data', models_dir='../models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
        self.scheduled_jobs = {}
        self.experiment_history = []
        
        # Ensure directories exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        
    def train_model(self, model_type='custom', data_file=None, epochs=10, batch_size=32, 
                    hyperparams=None, save_experiment=True):
        """
        Train a model with specified parameters
        """
        logger.info(f"Starting model training: {model_type}")
        
        # Set hyperparameters
        if hyperparams is None:
            hyperparams = {}
        
        # Default hyperparameters
        default_hyperparams = {
            'hidden_layers': [64, 32]
        }
        
        # Update with provided hyperparams
        default_hyperparams.update(hyperparams)
        
        # Start training
        start_time = time.time()
        
        try:
            model = self.trainer.train(
                model_type=model_type,
                data_file=data_file,
                epochs=epochs,
                batch_size=batch_size,
                hidden_layers=default_hyperparams['hidden_layers']
            )
            
            training_time = time.time() - start_time
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            logger.info(f"Model metrics: {self.trainer.metrics}")
            
            # Save experiment details
            if save_experiment:
                self._save_experiment(
                    model_type=model_type,
                    data_file=data_file,
                    epochs=epochs,
                    batch_size=batch_size,
                    hyperparams=default_hyperparams,
                    metrics=self.trainer.metrics,
                    training_time=training_time
                )
            
            return model, self.trainer.metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return None, None
    
    def _save_experiment(self, model_type, data_file, epochs, batch_size, 
                        hyperparams, metrics, training_time):
        """
        Save experiment details
        """
        experiment = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': model_type,
            'data_file': data_file,
            'epochs': epochs,
            'batch_size': batch_size,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'training_time': training_time
        }
        
        self.experiment_history.append(experiment)
        
        # Save experiment history
        experiment_file = os.path.join(self.models_dir, 'experiment_history.pkl')
        joblib.dump(self.experiment_history, experiment_file)
        
        # Also save as CSV for easier viewing
        history_df = pd.DataFrame(self.experiment_history)
        history_df.to_csv(os.path.join(self.models_dir, 'experiment_history.csv'), index=False)
        
        logger.info(f"Experiment saved: {experiment['timestamp']}")
    
    def load_experiment_history(self):
        """
        Load experiment history
        """
        experiment_file = os.path.join(self.models_dir, 'experiment_history.pkl')
        
        if os.path.exists(experiment_file):
            self.experiment_history = joblib.load(experiment_file)
            return self.experiment_history
        else:
            logger.info("No experiment history found")
            return []
    
    def plot_experiment_history(self, metric='accuracy'):
        """
        Plot the performance of models over different experiments
        """
        if not self.experiment_history:
            self.load_experiment_history()
        
        if not self.experiment_history:
            logger.warning("No experiment history to plot")
            return
        
        # Extract data
        timestamps = [exp['timestamp'] for exp in self.experiment_history]
        values = [exp['metrics'].get(metric, 0) for exp in self.experiment_history]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, 'o-', label=metric)
        plt.xlabel('Experiment Timestamp')
        plt.ylabel(metric.capitalize())
        plt.title(f'Model {metric.capitalize()} Over Experiments')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.models_dir, f'{metric}_history.png'))
        logger.info(f"Plot saved: {metric}_history.png")
        
        return plt
    
    def schedule_training(self, interval='daily', time='02:00', model_type='custom', 
                         data_file=None, epochs=10, batch_size=32, hyperparams=None):
        """
        Schedule automated model training
        """
        job_id = f"training_{model_type}_{interval}_{time}"
        
        def job():
            logger.info(f"Running scheduled training: {job_id}")
            self.train_model(
                model_type=model_type,
                data_file=data_file,
                epochs=epochs,
                batch_size=batch_size,
                hyperparams=hyperparams
            )
        
        if interval.lower() == 'daily':
            self.scheduled_jobs[job_id] = schedule.every().day.at(time).do(job)
        elif interval.lower() == 'weekly':
            self.scheduled_jobs[job_id] = schedule.every().week.at(time).do(job)
        elif interval.lower() == 'hourly':
            self.scheduled_jobs[job_id] = schedule.every().hour.do(job)
        else:
            raise ValueError(f"Unknown interval: {interval}")
        
        logger.info(f"Scheduled training job: {job_id}")
        return job_id
    
    def cancel_scheduled_job(self, job_id):
        """
        Cancel a scheduled job
        """
        if job_id in self.scheduled_jobs:
            schedule.cancel_job(self.scheduled_jobs[job_id])
            del self.scheduled_jobs[job_id]
            logger.info(f"Canceled scheduled job: {job_id}")
            return True
        else:
            logger.warning(f"Job not found: {job_id}")
            return False
    
    def run_scheduler(self, blocking=True):
        """
        Run the scheduler to execute scheduled jobs
        """
        if not self.scheduled_jobs:
            logger.warning("No jobs scheduled")
            return
        
        logger.info(f"Running scheduler with {len(self.scheduled_jobs)} jobs")
        
        if blocking:
            while True:
                schedule.run_pending()
                time.sleep(1)
        else:
            schedule.run_pending()
    
    def auto_experiment(self, model_types=None, epoch_options=None, batch_sizes=None):
        """
        Automatically run multiple experiments with different parameters
        """
        if model_types is None:
            model_types = ['custom', 'simple']
        
        if epoch_options is None:
            epoch_options = [5, 10, 20]
        
        if batch_sizes is None:
            batch_sizes = [16, 32, 64]
        
        results = []
        
        logger.info("Starting auto-experimentation")
        
        for model_type in model_types:
            for epochs in epoch_options:
                for batch_size in batch_sizes:
                    logger.info(f"Experiment: {model_type}, epochs={epochs}, batch_size={batch_size}")
                    
                    # Train model
                    model, metrics = self.train_model(
                        model_type=model_type,
                        epochs=epochs,
                        batch_size=batch_size,
                        save_experiment=True
                    )
                    
                    if metrics:
                        results.append({
                            'model_type': model_type,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'metrics': metrics
                        })
        
        # Find best model based on accuracy
        if results:
            best_experiment = max(results, key=lambda x: x['metrics']['accuracy'])
            logger.info(f"Best experiment: {best_experiment}")
            return best_experiment
        else:
            logger.warning("No successful experiments")
            return None

    def predict_batch(self, texts, model_type='custom'):
        """
        Make predictions on a batch of texts
        """
        try:
            return self.trainer.predict(texts, model_type=model_type)
        except FileNotFoundError:
            # If model doesn't exist yet, train a new one first
            logger.info("No trained model found. Training a new model first...")
            model, _ = self.train_model(model_type=model_type, epochs=5)
            if model is not None:
                # Now try prediction again
                return self.trainer.predict(texts, model_type=model_type)
            else:
                logger.error("Failed to train a model. Cannot make predictions.")
                return None

    def monitor_model_performance(self, data_file=None, interval='daily', time='03:00'):
        """
        Schedule regular model performance monitoring
        """
        job_id = f"monitor_performance_{interval}_{time}"
        
        def job():
            logger.info("Running model performance monitoring")
            
            # Load the latest model
            try:
                self.trainer.load_model()
                
                # Load test data or create sample data
                if data_file:
                    df = self.trainer.data_processor.load_data(data_file)
                else:
                    df = self.trainer.data_processor.create_sample_data(save=False)
                
                # Prepare data
                _, X_test, _, y_test = self.trainer.data_processor.prepare_data(
                    df, 'message', 'category'
                )
                
                # Evaluate
                metrics = self.trainer.evaluate(X_test, y_test)
                
                # Log metrics
                logger.info(f"Monitoring metrics: {metrics}")
                
                # Check if performance degraded
                threshold = 0.75  # example threshold
                if metrics['accuracy'] < threshold:
                    logger.warning(f"Model performance below threshold: {metrics['accuracy']} < {threshold}")
                    
                    # Could trigger retraining or alert here
                    
                return metrics
                
            except Exception as e:
                logger.error(f"Error during performance monitoring: {str(e)}")
                return None
        
        # Schedule the job
        if interval.lower() == 'daily':
            self.scheduled_jobs[job_id] = schedule.every().day.at(time).do(job)
        elif interval.lower() == 'weekly':
            self.scheduled_jobs[job_id] = schedule.every().week.at(time).do(job)
        elif interval.lower() == 'hourly':
            self.scheduled_jobs[job_id] = schedule.every().hour.do(job)
        else:
            raise ValueError(f"Unknown interval: {interval}")
        
        logger.info(f"Scheduled monitoring job: {job_id}")
        return job_id


if __name__ == "__main__":
    # Example usage
    automator = Automator()
    
    # Train a model
    model, metrics = automator.train_model(epochs=5)
    
    # Example prediction
    texts = [
        "Hello, can we meet tomorrow?",
        "URGENT: You've won a free prize! Claim now!"
    ]
    predictions = automator.predict_batch(texts)
    
    for text, pred in zip(texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {'SPAM' if pred == 1 else 'NOT SPAM'}")
        print()
    
    # Run auto-experimentation to find best model
    # automator.auto_experiment()
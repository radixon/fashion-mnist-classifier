import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import mlflow
import mlflow.pytorch
import numpy as np

# Add the src directory of the project
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, '..'))
# sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import modules from src directory
from data.data_loader import load_datasets, get_dataloaders
from data.preprocessing import get_transforms
from models.cnn_models import VanillaCNN, DeepCNN
from models.resnet_model import FashionResNet
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator, save_metrics
from evaluation.visualization import plot_confusion_matrix, plot_training_history
from utils.config import load_config
from utils.logger import setup_logging
from utils.helpers import get_timestamp_str, FASHION_MNIST_CLASSES

logger = logging.getLogger(__name__)

def main():
    """
    Training process function
    """
    print("----- Fashion MNIST Training Script -----")

    # Load Configuration Files
    config = load_config()
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    paths_config = config['paths']
    mlflow_config = config['mlflow']

    # Set MLflow Tracking URI and Experiment Name
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment_name'])

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Get run ID
        run_id = run.info.run_id

        # Setup Logging
        timestamp = get_timestamp_str()
        log_filename = paths_config['train_log_filename'].format(timestamp=timestamp)
        setup_logging(paths_config['logs_dir'], log_filename)
        logger.info(f"----- Fashion MNIST Training Script (MLflow Run ID: {run_id}) -----")

        # Log Parameters to MLflow
        mlflow.log_params(data_config)
        mlflow.log_params(training_config)
        mlflow.log_params(model_config)

        # Device Configuration
        device = torch.device("cuda" if torch.cuda.is_available() and training_config['device'] == "cuda" else "cpu")
        logger.info(f" Using Device: {device}")
        # Log Device Used
        mlflow.log_param("device", str(device))

        # Data Loading and Preprocessing
        logger.info("\nLoading datasets and creating DataLoaders...")
        transform = get_transforms()

        # Load datasets and apply transforms
        train_dataset, test_dataset = load_datasets(raw_data_path=data_config['raw_data_path'], transforms=transform, download=False)

        # Create DataLoaders
        train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=training_config['batch_size'], 
                                                num_workers=training_config['num_workers'], pin_memory=training_config['pin_memory'])

        # Model Initialization
        logger.info("\nInitializing model...")
        model_name = model_config['name']
        
        # Create an instance of VanillaCNN
        model = None
        if model_name == "VanillaCNN":
            model = VanillaCNN(input_dim=tuple(data_config['input_shape']), num_classes=data_config['num_classes'])
        elif model_name == "DeepCNN":
            model = DeepCNN(input_dim=tuple(data_config['input_shape']), num_classes=data_config['num_classes'], **model_config['deep_cnn_params'])
        elif model_name == "FashionResNet":
            model = FashionResNet(input_dim=tuple(data_config['input_shape']), num_classes=data_config['num_classes'], **model_config['fashion_resnet_params'])
        else:
            logger.error(f"Unknown Model Name: {model_name}")
            sys.exit(1)
        logger.info(f"Model Architecture: \n{model}") # Model Architecture
        
        # Loss Function and Optimizer
        # CrossEntropyLoss is used for multi-class classification
        logger.info("\nSettig up loss function and optimizer")
        if training_config['loss_function'] == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        else:
            logger.error(f"Unknown loss function: {training_config['loss_function']}")
            sys.exit(1)

        # Adam optimizer
        if training_config['optimizer'] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        elif training_config['optimizer'] == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=training_config['learning_rate'])
        else:
            logger.error(f"Unknown optimizer: {training_config['optimizer']}")
            sys.exit(1)

        # Model Trainer
        logger.info("===== Begin Training =====")

        # Create an instance of the ModelTrainer
        trainer = ModelTrainer(model, device)

        # Store History
        history = {'train_loss':[], 'train_accuracy':[], 'val_loss': [], 'val_accuracy': []}

        # Training Loop
        for epoch in range(1, training_config['epochs'] + 1):
            # Train for one epoch
            logger.info(f"\nEpoch {epoch} of {training_config['epochs']}")
            train_loss, train_accuracy = trainer.train_mode(train_loader, optimizer, criterion)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Minimum Viable Product validation using test_loader
            val_loss, val_accuracy = trainer.validate_mode(test_loader, criterion)
            logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            # Store for Plotting
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
        logger.info("\n===== Training Complete =====")

        # Save training model state dictionary
        model_save_path = paths_config['model_save_dir']
        temp_model_file = paths_config['temp_model_file']
        os.makedirs(model_save_path, exist_ok=True)
        full_model_save_path = os.path.join(model_save_path, temp_model_file)
        torch.save(model.state_dict(), full_model_save_path)
        logger.info(f"Model state dictionary saved to: {full_model_save_path}")

        # Perform Evaluation and Log Results
        logger.info("\nPerforming evaluation and logging results")
        evaluator = ModelEvaluator(model, device)
        true_labels, predicted_labels = evaluator.get_labels(test_loader)

        # Calculate Metrics
        metrics = evaluator.evaluate_model_performance(true_labels, predicted_labels, FASHION_MNIST_CLASSES)

        # Save Metrics
        metrics_filename = f"{paths_config['evaluation_metrics_filename_prefix']}{timestamp}.json"
        metrics_save_path_local = os.path.join(paths_config['metrics_save_dir'], metrics_filename)
        save_metrics(metrics, metrics_save_path_local)
        mlflow.log_artifact(metrics_save_path_local, "evaluation_results")

        # Plot Confusion Matrix
        conf_matrix = metrics["confusion_matrix"]
        conf_matrix_filename = f"{paths_config['confusion_matrix_filename_prefix']}{timestamp}.png"
        conf_matrix_save_path_local = os.path.join(paths_config['figures_save_dir'], conf_matrix_filename)
        plot_confusion_matrix(np.array(conf_matrix), FASHION_MNIST_CLASSES, conf_matrix_save_path_local, normalize=True, 
                              title=f'Confusion Matrix (Accuracy: {metrics["overall_accuracy"]:.4f})')
        
        mlflow.log_artifact(conf_matrix_save_path_local, "evaluation_results")

        # Plot Training History
        history_filename = f"training_history_{timestamp}.png"
        history_save_path_local = os.path.join(paths_config['figures_save_dir'], history_filename)
        plot_training_history(history, history_save_path_local, title=f"Training History for {model_config['name']}")
        mlflow.log_artifact(history_save_path_local, "training_results")
        logger.info(f"Accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info("Evaluation Results and Plots logged as MLflow artifacts")

    logger.info("===== MLflow run finished =====")


if __name__ == "__main__":
    try:
        import tqdm
    except ImportError:
        print("tqdm not found.  Install:  pip install tqdm")
        sys.exit(1)
    main()
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.data_loader import load_datasets, get_dataloaders
from data.preprocessing import get_transforms
from models.cnn_models import VanillaCNN, DeepCNN
from models.resnet_model import FashionResNet
from utils.config import load_config
from utils.logger import setup_logging
from utils.helpers import get_timestamp_str, FASHION_MNIST_CLASSES

def load_best_model(model_path: str, model_name: str, device: torch.device) -> nn.Module:
    """
    Load the best saved model from checkpoint.

    Args:
        model_path (str): Path to saved model checkpoint
        model_name (str): Name of the model architecture
        device (torch.device): Device to load the model onto
    
    Returns:
        nn.Module: Loaded PyTorch model in evaluation mode
    """
    logger = logging.getLogger(__name__)
    config = load_config()
    data_config = config['data']
    model_config = config['model']

    # Initialize the model
    if model_name == "VanillaCNN":
        model = VanillaCNN(tuple(data_config['input_shape']), data_config['num_classes'])
    elif model_name == "DeepCNN":
        model = DeepCNN(tuple(data_config['input_shape']), data_config['num_classes'], **model_config['deep_cnn_params'])
    elif model_name == "FashionResNet":
        model = FashionResNet(tuple(data_config['input_shape']), data_config['num_classes'], **model_config['fashion_resnet_params'])
    else:
        raise ValueError(f"Unknown Model Name: {model_name}")
    
    # Load Saved State Dict
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model Checkpoint not found at: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"Successfully loaded {model_name} from {model_path}")
    return model

def predict_and_visualize(model: nn.Module, test_loader: DataLoader, device: torch.device, num_samples: int=16) -> Tuple[plt.Figure, torch.Tensor, torch.Tensor]:
    """
    Make predictions and create visualization.

    Args:
        model (nn.Module): Trained PyTorch model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device for computation
        num_samples (int): Number of samples to visualize

    Returns:
        Tuple[plt.Figure, torch.Tensor, torch.Tensor, torch.Tensor]: Figure object, images, true labels, predicted labels
    """
    logger = logging.getLogger(__name__)

    # Get test data
    images, true_labels = next(iter(test_loader))

    # Get the number of prescribed samples
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]

    # Make Predictions
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        predicted_labels = torch.argmax(outputs, dim=1)

    # Move to CPU for visualization
    images = images.cpu()
    predicted_labels = predicted_labels.cpu()

    # Create Visualization
    rows = int(num_samples ** 0.5)
    cols = int((num_samples + rows - 1) // rows)
    fig, axes = plt.subplots(rows, cols, figsize=(12,12))
    axes = axes.flatten()
    for idx in range(num_samples):
        ax = axes[idx]

        # Display image
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Create title with true and predicted labels
        true_label = true_labels[idx].item()
        pred_label = predicted_labels[idx].item()

        true_class = FASHION_MNIST_CLASSES[true_label]
        pred_class = FASHION_MNIST_CLASSES[pred_label]

        # Color title
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {true_class} \nPred: {pred_class}'
        ax.set_title(title, color=color, fontsize=10)
    
    plt.suptitle('Sample Predictions', fontsize=16)
    plt.tight_layout()
    return fig, images, true_labels, predicted_labels


def calculate_sample_accuracy(true_labels: torch.Tensor, predicted_labels: torch.Tensor) -> Tuple[int, int, float]:
    """
    Calculate accuracy metrics for the sample.

    Args:
        true_labels (torch.Tensor): Ground truth labels
        predicted_labels (torch.Tensor): Model predictions
    
    Returns:
        Tuple[int, int, float]: Correct predictions, total predictions, accuracy
    """
    correct_predictions = (true_labels == predicted_labels).sum().item()
    total_predictions = len(true_labels)
    sample_accuracy = correct_predictions / total_predictions
    return correct_predictions, total_predictions, sample_accuracy


def save_prediction_results(fig: plt.Figure, save_path: str, model_name: str, sample_accuracy: float, logger: logging) -> None:
    """
    Save prediction visualization and log results

    Args:
        fig (plt.Figure): Matplotlib figure
        save_path (str): Path to save figure
        model_name (str): Name of the model used
        sample_accuracy (float): Calculated sample accuracy
        logger (logging): Logger instance for output
    
        Returns:
            None
    """
    # Verify directory existance
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Predction Visualization saved to: {save_path}")
    logger.info(f"Model: {model_name}, Sample accuracy: {sample_accuracy:.4f}")


def main() -> None:
    """
    Main prediction function.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Fashion MNIST prediction script")

    # Load configuration
    config = load_config()
    model_config = config['model']
    paths_config = config['paths']
    training_config = config['training']
    data_config = config['data']

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logger.info(f"Using Device: {device}")

    # Load test data
    logger.info("Loading Test Data")
    transform = get_transforms()
    train_dataset, test_dataset = load_datasets(raw_data_path=data_config['raw_data_path'], transforms=transform, download=True)

    # Create DataLoaders
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset, batch_size=training_config['batch_size'], 
                                            num_workers=training_config['num_workers'], pin_memory=training_config['pin_memory'])
    
    # Model Path
    model_path = os.path.join(paths_config['model_save_dir'], paths_config['best_model_filename'])
    model_name = model_config['name']

    try:
        # Load Best Model
        model = load_best_model(model_path, model_name, device)

        # Make predictions and create visualization
        logger.info("Making predictions on sample test images")
        fig, images, true_labels, predicted_labels = predict_and_visualize(model, test_loader, device)

        # Calculate Metrics
        correct_predictions, total_predictions, sample_accuracy = calculate_sample_accuracy(true_labels, predicted_labels)

        # Save Results
        timestamp = get_timestamp_str()
        save_filename = f"sample_predictions_script_{model_name}_{timestamp}.png"
        save_path = os.path.join(paths_config['figures_save_dir'], save_filename)
        save_prediction_results(fig, save_path, model_name, sample_accuracy, logger)

        # Display Results
        logger.info(f"Sample accuracy: {correct_predictions} / {total_predictions} = {sample_accuracy:.4f}")

        # Show Plot
        plt.show()

        logger.info("Prediction script successfully completed")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load model: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        return

if __name__ == "__main__":
    main()
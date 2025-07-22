import os
import sys
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.data_loader import load_datasets, dataloaders
from data.preprocessing import get_transforms
from models.cnn_models import VanillaCNN
from evaluation.evaluator import ModelEvaluator, save_metrics
from evaluation.visualization import plot_confusion_matrix

# Configuration Constants
RAW_DATA_PATH = 'data/raw'
BATCH_SIZE = 64
NUM_CLASSES = 10
INPUT_DIM = (1, 28, 28)
MODEL_PATH = 'vanilla_cnn_model.pth'

# Fashion MNIST output class names
FASHION_MNIST_CLASSES = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

def main():
    print("----- Fashion MNIST Evaluation Script -----")

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Data Loading
    print("\nLoading test dataset...")
    transform = get_transforms(train=False) # Use Validation Transforms
    _, test_dataset = load_datasets(raw_data_path=RAW_DATA_PATH, transforms=transform)
    _, test_loader = dataloaders(train_dataset=None, test_dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=0)  

    # Model Loading
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run scripts/train.py and save the model")
        sys.exit(1)
    
    model = VanillaCNN(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH,  map_location=device)) # Load state_dict into device
    print("Loaded Model Successfully")

    # Model Evaluation
    print("\nStarting model evaluation...")
    evaluator = ModelEvaluator(model=model, device=device)
    true_labels, predicted_labels = evaluator.get_labels(test_loader)

    # Calculate Metrics
    metrics = evaluator.evaluate_model_performance(true_labels=true_labels, predicted_labels=predicted_labels, class_names=FASHION_MNIST_CLASSES)

    # Save Metrics
    timestape = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_save_path = os.path.join('results', 'metrics', f'evaluation_metrics_{timestape}.json')
    save_metrics(metrics=metrics, save_path=metrics_save_path)

    # Visualize Results
    conf_matrix = metrics["confusion_matrix"]
    conf_matrix_save_path = os.path.join('results', 'figures', f'confusion_matrix_{timestape}.png')
    plot_confusion_matrix(cm=np.array(conf_matrix), class_names=FASHION_MNIST_CLASSES, save_path=conf_matrix_save_path, normalize=True, 
                            title=f'Confusion Matrix (Accuracy: {metrics["overall_accuracy"]:.4f})')
    
    print("\n----- Evaluation Complete -----")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print("\nClassification Report:\n")
    for class_name, class_metric in metrics['classification_report'].items():
        if isinstance(class_metric, dict):
            print(f" {class_name}: Precision={class_metric['precision']:.2f}, "
                  f"Recall={class_metric['recall']:.2f}, F1-Score={class_metric['f1-score']:.2f}, "
                  f"Support={class_metric['support']}")
        else:
            print(f" {class_name}: {class_metric:.2f}" if isinstance(class_metric, float) else f" {class_name}: {class_metric}")


if __name__ == "__main__":
    try:
        import numpy as np
        import sklearn
        import torch
        import torch.nn
    except ImportError:
        print("Error: Libraries are required. Install libraries: pip install numpy scikit-learn torch")
        sys.exit(1)
    main()
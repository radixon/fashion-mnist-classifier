import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.data_loader import load_datasets, dataloaders
from data.preprocessing import get_transforms
from models.cnn_models import VanillaCNN
from evaluation.evaluator import ModelEvaluator, save_metrics
from evaluation.visualization import plot_confusion_matrix
from utils.config import load_config

# Fashion MNIST output class names
FASHION_MNIST_CLASSES = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

def main():
    print("----- Fashion MNIST Evaluation Script -----")

    # Load Configuration
    config = load_config()
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    paths_config = config['paths']

    # Device Configuration
    device = torch.device("cuda" if training_config['device'] == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Data Loading
    print("\nLoading test dataset...")
    transform = get_transforms(train=False) # Use Validation Transforms
    _, test_dataset = load_datasets(raw_data_path=data_config['raw_data_path'], transforms=transform)
    _, test_loader = dataloaders(train_dataset=None, test_dataset=test_dataset, batch_size=training_config['batch_size'], 
                                 num_workers=training_config['num_workers'], pin_memory=training_config['pin_memory'])  

    # Model Loading
    model_name = model_config['name']
    model = None
    if model_name == "VanillaCNN":
        model = VanillaCNN(input_dim=tuple(data_config['input_shape']), num_classes=data_config['num_classes'])
    else:
        raise VAlueError(f"Unknown model name: {model_name}") 
    
    model_save_dir = paths_config['model_save_dir']
    temp_model_file = paths_config['temp_model_file']
    full_model_load_path = os.path.join(model_save_dir, temp_model_file)

    print(f"\nLoading model from {full_model_load_path}...")
    if not os.path.exists(full_model_load_path):
        print(f"Error: Model not found at {full_model_load_path}")
        sys.exit(1)
    
    model.load_state_dict(torch.load(full_model_load_path, map_location=device))
    print("Model successfully loaded")

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
        print("Error: Libraries are missing. Install libraries: pip install numpy scikit-learn torch")
        sys.exit(1)
    main()
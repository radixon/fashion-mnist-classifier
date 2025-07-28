import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add the src directory of the project
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, '..'))
# sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import modules from src directory
from data.data_loader import load_datasets, dataloaders
from data.preprocessing import get_transforms
from models.cnn_models import VanillaCNN
from training.trainer import ModelTrainer
from utils.config import load_config


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

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() and training_config['device'] == "cuda" else "cpu")
    print(f" Using Device: {device}")

    # Data Loading and Preprocessing
    # Default transforms
    print("\nLoading datasets and creating DataLoaders...")
    transform = get_transforms()

    # Load datasets and apply transforms
    train_dataset, test_dataset = load_datasets(raw_data_path=data_config['raw_data_path'], transforms=transform)

    # Create DataLoaders
    train_loader, test_loader = dataloaders(train_dataset, test_dataset, batch_size=training_config['batch_size'], 
                                            num_workers=training_config['num_workers'], pin_memory=training_config['pin_memory'])

    # Model Initialization
    print("\nInitializing model...")
    model_name = model_config['name']
    
    # Create an instance of VanillaCNN
    model = None
    if model_name == "VanillaCNN":
        model = VanillaCNN(input_dim=tuple(data_config['input_shape']), num_classes=data_config['num_classes'])
    else:
        raise ValueError(f"Unknown Model Name: {model_name}")
    print(model) # Model Architecture
    
    # Loss Function and Optimizer
    # CrossEntropyLoss is used for multi-class classification
    if training_config['loss_function'] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {training_config['loss_function']}")

    # Adam optimizer
    if training_config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    elif training_config['optimizer'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=training_config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")

    # Model Trainer
    print("Begin training...")

    # Create an instance of the ModelTrainer
    trainer = ModelTrainer(model, device)

    # Training Loop
    for epoch in range(1, training_config['epochs'] + 1):
        # Train for one epoch
        train_loss, train_accuracy = trainer.train_mode(train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Minimum Viable Product validation using test_loader
        val_loss, val_accuracy = trainer.validate_mode(test_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print("\n===== Training Complete =====")

    # Save training model state dictionary
    model_save_path = paths_config['model_save_dir']
    temp_model_file = paths_config['temp_model_file']
    os.makedirs(model_save_path, exist_ok=True)
    full_model_save_path = os.path.join(model_save_path, temp_model_file)
    torch.save(model.state_dict(), full_model_save_path)
    print(f"Model state dictionary saved to: {full_model_save_path}")


if __name__ == "__main__":
    try:
        import tqdm
    except ImportError:
        print("tqdm not found.  Install:  pip install tqdm")
        sys.exit(1)
    main()
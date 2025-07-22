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


# Constants (to be in config.yaml)
RAW_DATA_PATH = 'data/raw'
BATCH_SIZE = 64             # Samples per batch
LEARNING_RATE = 0.001
EPOCHS = 10                 
NUM_CLASSES = 10
INPUT_DIM = (1, 28, 28)

def main():
    """
    Training process function
    """
    print("----- Fashion MNIST Training Script -----")

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using Device: {device}")

    # Data Loading and Preprocessing
    # Default transforms
    transform = get_transforms()

    # Load datasets and apply transforms
    train_dataset, test_dataset = load_datasets(raw_data_path=RAW_DATA_PATH, transforms=transform)

    # Create DataLoaders
    train_loader, test_loader = dataloaders(train_dataset, test_dataset, BATCH_SIZE)

    # Model Initialization
    print("Initializing VanillaCNN model...")

    # Create an instance of VanillaCNN
    model = VanillaCNN(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    print(model) # Model Architecture

    # Loss Function and Optimizer
    # CrossEntropyLoss is used for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Model Trainer
    print("Begin training...")

    # Create an instance of the ModelTrainer
    trainer = ModelTrainer(model, device)

    # Training Loop
    for epoch in range(1, EPOCHS + 1):
        # Train for one epoch
        train_loss, train_accuracy = trainer.train_mode(train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Minimum Viable Product validation using test_loader
        val_loss, val_accuracy = trainer.validate_mode(test_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print("\n===== Training Complete =====")

    # Save training model state dictionary
    model_save_path = 'vanilla_cnn_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model state dictionary saved to: {model_save_path}")


if __name__ == "__main__":
    try:
        import tqdm
    except ImportError:
        print("tqdm not found.  Install:  pip install tqdm")
        sys.exit(1)
    main()
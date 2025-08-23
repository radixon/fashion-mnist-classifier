import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))
from training.trainer import ModelTrainer
from models.cnn_models import VanillaCNN
from data.data_loader import get_dataloaders, load_datasets

class TestTraining:
    """
    Test suite for training functionality.
    """
    @pytest.fixture
    def sample_model(self) -> VanillaCNN:
        """
        Create a model for testing.

        Returns:
            VanillaCNN: An instance of the VanillaCNN model
        """
        return VanillaCNN(input_dim=(1,28,28), num_classes=10)
    
    @pytest.fixture
    def device(self) -> torch.device:
        """
        Get available device.

        Returns:
            torch.device:  CPU device for consistent testing
        """
        return torch.device("cpu")
    
    @pytest.fixture
    def sample_data(self) -> torch.utils.data.DataLoader:
        """
        Create sample data for testing.

        Returns:
            torch.utils.data.DataLoader:  DataLoader for testing
        """
        images = torch.randn(8, 1, 28, 28)
        labels = torch.randint(0, 10, (8,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        return loader
    
    @pytest.fixture
    def trainer_setup(self, sample_model: VanillaCNN, device: torch.device) -> ModelTrainer:
        """
        Setup trainer.

        Args:
            sample_model (VanillaCNN): An instance of the VanillaCNN model
            device (torch.device): Device from instance
        
        Returns:
            ModelTrainer:  An instance of the configured trainer
        """
        return ModelTrainer(sample_model, device)
    
    def test_trainer_initialization(self, sample_model: VanillaCNN, device: torch.device):
        """
        Test ModelTrainer initialization.

        Args:
            sample_model (VanillaCNN): An instance of the VanillaCNN model
            device (torch.device): Device from instance
        """
        trainer = ModelTrainer(sample_model, device)
        assert trainer.model is sample_model, "Trainer should store the model"
        assert trainer.device == device, "Trainer should store the device"
        assert next(trainer.model.parameters()).device == device, "Model should move to device"

    def test_train_mode_returns_correct_types(self, trainer_setup: ModelTrainer, sample_data: torch.utils.data.DataLoader):
        """
        Verify train_mode returns loss and accuracy as floats.

        Args:
            trainer_setup (ModelTrainer): An instance of ModelTrainer
            sample_data (torch.utils.data.DataLoader): Sample data from instance
        """
        optimizer = optim.SGD(trainer_setup.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = trainer_setup.train_mode(sample_data, optimizer, criterion)

        assert isinstance(loss, float), "Training loss should be a float"
        assert isinstance(accuracy, float), "Training accuracy should be a float"
        assert 0.0 <= accuracy <= 1.0, "Accuracy should in range [0.0, 1.0]"
        assert loss >= 0.0, "Loss should be non-negative"
    
    def test_validate_mode_returns_correct_types(self, trainer_setup: ModelTrainer, sample_data: torch.utils.data.DataLoader):
        """
        Test validate_mode returns loss and accuracy as floats.

        Args:
            trainer_setup (ModelTrainer): An instance of ModelTrainer
            sample_data (torch.utils.data.DataLoader): Sample data from instance
        """
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = trainer_setup.validate_mode(sample_data, criterion)

        assert isinstance(loss, float), "Validation loss should be a float"
        assert isinstance(accuracy, float), "Validation accuracy should be a float"
        assert 0.0 <= accuracy <= 1.0, "Accuracy should be in range [0.0, 1.0]"
        assert loss >= 0.0, "Loss should be non-negative"
    
    def test_train_mode_updates_parameters(self, trainer_setup: ModelTrainer, sample_data: torch.utils.data.DataLoader):
        """
        Verify train_mode updates model parameters.

        Args:
            trainer_setup (ModelTrainer): An instance of ModelTrainer
            sample_data (torch.utils.data.DataLoader): Sample data from instance
        """
        optimizer = optim.SGD(trainer_setup.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Store initial parameters
        initial_parameters = [param.clone() for param in trainer_setup.model.parameters()]

        # Loop through one training epoch
        trainer_setup.train_mode(sample_data, optimizer, criterion)

        # Verify parameters have changed
        parameters_changed = False
        for initial, current in zip(initial_parameters, trainer_setup.model.parameters()):
            if not torch.equal(initial, current):
                parameters_changed = True
                break

        assert parameters_changed, "Model parameters should update after training"
    
    def test_model_mode_switching(self, trainer_setup: ModelTrainer, sample_data: torch.utils.data.DataLoader):
        """
        Verify trainer switches model modes correctly.

        Args:
            trainer_setup (ModelTrainer): An instance of ModelTrainer
            sample_data (torch.utils.data.DataLoader): Sample data from instance
        """
        optimizer = optim.SGD(trainer_setup.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Test training mode
        trainer_setup.train_mode(sample_data, optimizer, criterion)

        # Test validation mode
        trainer_setup.validate_mode(sample_data, criterion)

        # If no errors are raised this test passes
        assert True, "Model mode switching"
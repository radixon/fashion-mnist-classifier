import pytest
import torch
import os
import sys
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data_loader import get_dataloaders, load_datasets
from src.data.preprocessing import get_transforms

def get_data() -> Tuple[Dataset, Dataset]:
    """
    Get datasets necessary to test DataLoaders
    """
    transform = get_transforms()
    train_dataset, test_dataset = load_datasets(raw_data_path='../data/raw/FashionMNIST/raw', transforms=transform, download=True)
    return [train_dataset, test_dataset]

class TestDataLoader:
    """
    Test suite for data loading functionality.
    """

    def test_get_transforms(self):
        transform = get_transforms()

        # Verify transform is a Compose object
        assert hasattr(transform, 'transforms'), "Transform should be a torchvision.transforms.Compose object"

        # Verify transforms are as expected
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert 'ToTensor' in transform_types, "Transform should include ToTensor"
        assert 'Normalize' in transform_types, "Transform should include Normalize"
    

    def test_get_dataloaders_returns_correct_types(self):
        """
        Test get_dataloaders returns DataLoader objects
        """
        train, test = get_data()
        train_loader, test_loader = get_dataloaders(train, test, batch_size=64)

        assert isinstance(train_loader, DataLoader), "train_loader should be a DataLoader"
        assert isinstance(test_loader, DataLoader), "test_loader should be a DataLoader"
    
    def test_get_dataloaders_batch_size(self):
        """
        Test that DataLoaders works with specified batch size.
        """
        train, test = get_data()
        batch_size = 32
        train_loader, test_loader = get_dataloaders(train, test, batch_size=batch_size)

        # Get batch from each DataLoader
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        # Check batch sizes
        assert train_batch[0].shape[0] == batch_size, f"Train batch size should be {batch_size}"
        assert test_batch[0].shape[0] == batch_size, f"Test batch size should be {batch_size}"
    
    def test_data_shapes(self):
        """
        Test that data has the correct shapes.
        """
        train, test = get_data()
        train_loader, test_loader = get_dataloaders(train, test, batch_size=64)

        # Get batch
        images, labels = next(iter(train_loader))

        # Check image shape (batch_size, channels, height, width)
        assert images.shape == (64, 1, 28, 28), f"Image shape should be (64, 1, 28, 28), got {images.shape}"

        # Check labels shape: (batch_size,)
        assert labels.shape == (64,), f"Labels shape should be (64,), got {labels.shape}"

        # Check data types
        assert images.dtype == torch.float32, f"Images should be float32, got {images.dtype}"
        assert labels.dtype == torch.int64, f"Labels should be int64, got {labels.dtype}"
    
    def test_data_normalization(self):
        """
        Test that images are properly normalized.
        """
        train, _ = get_data()
        train_loader, _ = get_dataloaders(train, None, batch_size=64)
        images, _ = next(iter(train_loader))

        # After normalization, pixel values should be in the range [-1, 1]
        assert images.min() < 0, "Normalized images should have negative values"
        assert images.max() <= 1, "Normalized images should not have positive values with high magnitudes"
        assert images.min() >= -1, "Normalized images should not have negative values with high magnitudes"

    def test_different_batch_sizes(self):
        """
        Test that different batch sizes behave normally
        """
        train, test = get_data()
        for batch_size in [16,32,64,128]:
            train_loader, test_loader = get_dataloaders(train, test, batch_size=batch_size)
            
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))

            assert train_batch[0].shape[0] <= batch_size, "The batch size is smaller than the number of samples, this required DEBUGGING"
            assert test_batch[0].shape[0] <= batch_size, "The batch size is smaller than the number of samples, this required DEBUGGING"
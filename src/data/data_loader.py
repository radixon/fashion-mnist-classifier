import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def load_data(directory: str, download: bool):
    """
    Download Fashion MNIST dataset.

    Arguments:
        directory (str): Directory to save data
        download (bool): Whether to download if not present

    Returns:
        train_data, test_data (torchvision.datasets): Train and test datasets
    """

    xfm = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root=directory, train=True,transform=xfm, download=download)
    test_data = datasets.FashionMNIST(root=directory, train=False, transform=xfm, download=download)

    return train_data, test_data

def dataloaders(train_data: torch.Tensor, test_data: torch.Tensor, batch_size: int):
    """
    Create DataLoaders using datasets.

    Arguments:
        train_data: torch.utils.data.Dataset
        test_data: torch.utils.data.Dataset
        batch_size (int): Batch size
    
    Returns:
        train_loader, test_loader
    """

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
import torch
from torchvision import datasets                    # Required to load FashionMNIST dataset
from torch.utils.data import DataLoader, Dataset    # For creating data pipeline
import os
from typing import Tuple, Callable
from src.data.preprocessing import get_transforms   # Import project defined preprocessing functions

def load_datasets(raw_data_path: str = './data/raw', transforms: Callable=None, download: bool=True) -> Tuple[Dataset, Dataset]:
    """
    Loads Fashion MNIST training/test datasets
    If datasets aren't local, torchvision will download datasets.

    Args:
        raw_data_path (str):    The directory where FashionMNIST dataset will be stored
        xfm (Callable):         A torchvision.transforms.Compose object to apply to images.
                                Default transforms are 'get_transform'
        download (bool):        Download datasets if not found in 'raw_data_path'

    Returns:
        Tuple[Dataset, Dataset] (torchvision.datasets): A tuple containing train and test datasets
    """
    # Verify raw data path exists
    os.makedirs(raw_data_path, exist_ok=True)

    if transforms == None:
        train_transforms = get_transforms(train = True)
        test_transforms = get_transforms(train=False)
    else:
        # Use provided transform
        train_transforms = transforms
        test_transforms = transforms
    
    # Load the train dataset
    train_data = datasets.FashionMNIST(root=raw_data_path, train=True,transform=train_transforms, download=download)

    # Load the test dataset
    test_data = datasets.FashionMNIST(root=raw_data_path, train=False, transform=test_transforms, download=download)

    print(f"Fashion MNIST datasets successfully loaded.  Train Samples: {len(train_data)} \t Test Samples: {len(test_data)}")
    return train_data, test_data




def dataloaders(train_dataset: Dataset, test_dataset: Dataset, batch_size: int, num_workers: int=0, pin_memory: bool=True):
    """
    Create DataLoaders for given datasets.  DataLoaders are responsible for iterating over the dataset

    Args:
        train_data (Dataset):   The training dataset
        test_data (Dataset):    The testing dataset
        batch_size (int):       Samples per batch
        num_workers (int):      Subprocesses used for data loading.
                                0 tells the data to load in the main process
                                >0 speeds up loading using multi-core CPUs
        pin_memory (bool):      Speeds up data transfer to GPU by pinning memory
    
    Returns:
        Tuple[DataLoader, DataLoader]:  A tuple containing train/test dataloaders
    """

    # DataLoader:  train set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    # DataLoader:  test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"DataLoaders created with batch size: {batch_size}")
    return train_loader, test_loader
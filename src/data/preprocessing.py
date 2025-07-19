from torchvision import transforms
from typing import Callable

"""
Constants standard mean and standard deviation for images normalized to [0.0, 1.0]

Typically calculated from the training data.
"""
MNIST_mean = 0.5    # Mean for grayscale
MNIST_std = 0.5     # Standard deviation for grayscale


def get_transforms(train: bool=True) -> Callable:
    """
    Returns a torchvision.transforms.Compose object.

    Args:
        train (bool):   If True, returns transforms for training.
                        If False, returns transforms for validation/testing.
    
    Returns:
        Callable:       A torch.vision.transforms.Compose object that can be passed
                        to torchvision.datasets
    """
    transform_list = []
    
    if train:
        #  Training data augmentation
        pass
    
    # Convert image to a PyTorch Tensor
    # Converts pixel values from [0, 255] to [0.0, 1.0]
    # Changes image dimension order from (height, width, channels) to PyTorch's preferred (channels, height, width)
    transform_list.append(transforms.ToTensor())


    # Normalize the tensor with the dataset's mean and standard deviation
    # Transforms pixel values from [0.0, 1.0] to [-1.0, 1.0] given (mean, std) == (0.5, 0.5)
    transform_list.append(transforms.Normalize(MNIST_mean, MNIST_std))

    # Create a pipeline of transforms in a single callable object
    return transforms.Compose(transform_list)
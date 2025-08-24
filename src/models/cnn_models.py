import torch.nn as nn
from typing import Tuple
from .base_model import BaseModel   # Import BaseModel


class VanillaCNN(BaseModel):
    """
    A Convolutional Neural Network for Fashion MNIST classification.  Inherits from BaseModel
    and implements the build_model_ method define the layers of VanillaCNN
    """
    def __init__(self, input_dim: Tuple[int, int, int], num_classes: int):
        """
        Initializes VanillaCNN

        Args:
            input_dim (Tuple[int, int, int]):  Expected dimensions (channel, height, width) == (1, 28, 28) for Fashion MNIST
            num_classes (int): Number of classes in the output layer
        """
        super().__init__(input_dim, num_classes)

    def build_model_(self) -> nn.Module:
        """
        Define the layers of the VanillaCNN
        """
        channels, height, width = self.input_dim
        model = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),     # Output size: 32x28x28
            nn.ReLU(),  # Activation Function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 32x14x14 (reduces spatial dimensions)

            # Second Convolutional Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),   # Output size: 64x14x14
            nn.ReLU(),  # Activation Function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 64x7x7

            # Flatten the output for the fully connected layers
            nn.Flatten(),

            # Fully connected (Linear) Layers
            nn.Linear(64 * 7 * 7, 128),  # Fully connected layer
            nn.ReLU(),   # Activation Function
            nn.Linear(128, self.num_classes)  # Output layer goes from 128 features to 10 classes
        )
        return model


class DeepCNN(BaseModel):
    """
    A Convolutional Neural Network (CNN) for Fashion MNIST classification, with Batch Normalization and Dropout.
    """
    def __init__(self, input_dim: Tuple[int, int, int],
                 num_classes: int, conv1_out_channels: int = 64,
                 conv2_out_channels: int = 128,
                 conv3_out_channels: int = 256,
                 linear1_features: int = 512, dropout_rate: float = 0.3) -> None:
        """
        CNN Initialization
        Args:
            input_dim (Tuple[int, int, int]): Expected dimensions (channel, height, width) == (1, 28, 28) for Fashion MNIST
            num_classes (int): Number of classes in the output layer
            conv1_out_channels (int): Number of output channels from layer 1
            conv2_out_channels (int): Number of output channels from layer 2
            conv3_out_channels (int): Number of output channels from layer 3
            linear1_features (int): Number of features for the first linear layer
            dropout_rate (float): Node dropout probability for regularization
        """
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv3_out_channels = conv3_out_channels
        self.linear1_features = linear1_features
        self.dropout_rate = dropout_rate
        super().__init__(input_dim, num_classes)

    def build_model_(self) -> nn.Module:
        """
        Defines the layers of the DeepCNN model
        """
        C, H, W = self.input_dim
        h = H // (2 * 2)
        w = W // (2 * 2)
        flattended_features = self.conv3_out_channels * h * w

        model = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(C, self.conv1_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second Block
            nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third Convolutional Block
            nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.conv3_out_channels),
            nn.ReLU(),

            # Flatten the output
            nn.Flatten(),

            # Connected Layers
            nn.Linear(flattended_features, self.linear1_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.linear1_features, self.num_classes)
        )
        return model

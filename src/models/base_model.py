import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseModel(nn.Module, ABC):
    """
    Abstract base class that ensures models inherit from nn.Module and forward method.
    The design forces implementation of a build_model_ method.
    """
    def __init__(self, input_dim: Tuple[int, int, int],
                 num_classes: int) -> None:
        """
        Initialize BaseModel

        Args:
            input_dim (Tuple[int, int, int]):  Expected input shape (channels, height, weight) for PyTorch
            num_classes (int): Number of classes in the output layer
        """
        super().__init__()                  # Call the constructor of nn.Module
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self.build_model_()    # The CNN architecture is in build_model_()

    @abstractmethod  # Forces an error at instantiation if a method isn't in the design
    def build_model_(self) -> nn.Module:
        """
        This method should define and return the nn.Module that comprises the neural network's layers.
        """
        pass    # This is an abstract class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.  The build_model_() method is private, so this method passes the input
        through the 'model' attribute which equals build_model_().
        """
        x = self.model(x)
        return x

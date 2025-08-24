import torch
import torch.nn as nn
from typing import Type, List, Tuple

from .base_model import BaseModel


class BasicBlock(nn.Module):
    """
    A ResNet BasicBlock
    Consists of two 3x3 convolutional layers with Batch Normalization and ReLU.
    Includes a shortcut connection for residual learning.
    """
    expansion: int = 1    # Factor which the output channels expand

    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int = 1) -> None:
        super().__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for residual learning
        self.shortcut = nn.Sequential()
        # If dimensions change, use a 1x1 convolution to match changes
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, self.expansion * out_channels,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for shortcut connection
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut connection to the output
        out += identity
        out = self.relu(out)

        return out


class FashionResNet(BaseModel):
    """
    Simplified ResNet model adapted for Fashion MNIST
    Inherits from BaseModel
    """
    def __init__(self, input_dim: Tuple[int, int, int],
                 num_classes: int,
                 block_type: str = "BasicBlock",
                 num_blocks: List[int] = [2, 2, 2],
                 base_channels: int = 64) -> None:
        """
        Initializes the FashionResNet model

        Args:
            input_dim (Tuple[int, int, int]):   Expected input shape (channel, height, weight)
            num_classes (int):  Number of output classes
            block_type (str):   String name of the block type
            num_blocks (List[int]): List specifying the number of blocks in each of the 3 stages
            base_channels (int):    Initial number of output channels after the first convolution
        """
        self.block_class = self.get_block_class_(block_type)
        self.num_blocks_per_stage = num_blocks
        self.base_channels = base_channels

        # Initial input channels for the first _make_layer call
        self.in_channels = self.base_channels

        super().__init__(input_dim, num_classes)

    def get_block_class_(self, block_type_str: str) -> Type[nn.Module]:
        """
        Helper to get the block class from a string name

        Args:
            block_type (str):   String name of the block type
        """
        if block_type_str == "BasicBlock":
            return BasicBlock
        else:
            raise ValueError(f"Unsupported block type: {block_type_str}")

    def make_layer_(self, block: Type[nn.Module],
                    out_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        """
        Helper to create a ResNet stage
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))

            # Update in_channels for the next block
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def build_model_(self) -> nn.Module:
        """
        Defines the layers of the FashionResNet model
        """
        # Unpack input dimensions
        C, H, W = self.input_dim
        layers = []

        # 28x28 images, 3x3 conv with stride 1
        layers.append(nn.Conv2d(C, self.base_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(self.base_channels))
        layers.append(nn.ReLU(inplace=True))

        # Stage 1: No downsampling (stride = 1)
        layers.append(self.make_layer_(self.block_class, self.base_channels, self.num_blocks_per_stage[0], stride=1))

        # Stage 2: Downsamples by 2 (stride = 2)
        layers.append(self.make_layer_(self.block_class, self.base_channels * 2, self.num_blocks_per_stage[1], stride=2))

        # Stage 3: Downsamples by 2 (stride = 2)
        layers.append(self.make_layer_(self.block_class, self.base_channels * 4, self.num_blocks_per_stage[2], stride=2))

        # Average Pooling adapts to any input spatial size, outputs (Batch, Channels, 1, 1)
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Flatten followed by Linear Layer
        layers.append(nn.Flatten())
        features = self.base_channels * 4 * self.block_class.expansion
        layers.append(nn.Linear(features, self.num_classes))

        return nn.Sequential(*layers)

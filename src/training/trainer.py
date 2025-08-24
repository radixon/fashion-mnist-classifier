import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # Visually track progress during training/validation
from typing import Tuple


class ModelTrainer:
    """
    Manages the training and validation process.

    Orchestrates the forward/backward passes and optimization steps.
    """
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """
        Initializes ModelTrainer

        Args:
            model (nn.Module):  Instance of a class that inherits from nn.Module
            device (torch.device): Either GPU or CPU where the model and data will be processed
        """
        self.model = model.to(device)
        self.device = device

    def train_mode(self, dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Performs one full training epoch over the provided dataloader.

        Args:
            dataloader (DataLoader):  DataLoader provides training data in batches.
            optimizer (torch.optim.Optimizer):  The optimizer used to update model weights.
            criterion (nn.Module):  The loss function used to calculate the training loss

        Returns:
            Tuple[float, float]:  Average training loss and average training accuracy
        """
        self.model.train()  # Training mode allows active Dropout, BatchNorm updates, etc.
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over the training data batches
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass for the current batch
            yhat = self.model(inputs)

            # Calculate the loss
            loss = criterion(yhat, labels)

            # Backward pass that compute gradients of the loss w.r.t. model parameters
            loss.backward()

            # Optimizer step that updates model weights boased on gradients
            optimizer.step()

            # Accumulate loss and calculate accuracy for the current epoch
            running_loss += loss.item() * inputs.size(0)

            # Get the class with the highest probability
            _, predicted = torch.max(yhat.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()   # Count accurate predictions

        # Calculate average loss
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

    def validate_mode(self, dataloader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float]:
        """
        Performs one full validation epoch over the provided dataloader.  No gradient calculations
        are performed during validation.

        Args:
            dataloader (DataLoader):  DataLoader provides validation data in batches.
            criterion (nn.Module):  The loss function used during validation.

        Returns:
            Tuple[float, float]:  Average validation loss and average validation accuracy.
        """
        self.model.eval()  # Set model to evaluation mode inactivates Dropout, and BatchNorm update
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Disable gradient calculation
        with torch.no_grad():
            # Iterate over the validation batches
            for inputs, labels in tqdm(dataloader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                yhat = self.model(inputs)

                # Calculate the loss
                loss = criterion(yhat, labels)

                # Accumulate loss and calulate accuracy
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(yhat.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()   # Count accurate predictions

        # Calculate average loss and average accuracy for current epoch
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        return epoch_loss, epoch_accuracy

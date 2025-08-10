import os
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Stops training if validation metric does not improve after give patience quantity.
    """
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 5, min_delta: float = 0.001):
        """
        Args:
            monitor (str):  Metric to monitor
            mode (str): One of 'min' or 'max'.  In 'min' mode, training will stop when the quantity monitored has
                        stopped decreasing.  In 'max' mode training will stop when the quantity monitored has
                        stopped increasing.
            patience (int): Number of epochs to wait for required level of improvement before stopping.
            min_delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epochs_no_improve = 0
        self.stop_training = False
    
    def __call__(self, current_value: float):
        """
        Call this method after each validation epoch
        """
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
        elif self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
        
        if self.epochs_no_improve == self.patience:
            self.stop_training = True
            logger.info(f"Early Stopping Triggered!!! No improvement in '{self.monitor}' for {self.patience} epochs.")


class ModelCheckpoint:
    """
    Saves the model's state_dict when the monitored metric improves
    """
    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = True):
        """
        Args:
            filepath (str): Path to save the model.
            monitor (str): Metric to monitor
            mode (str): One of 'min' or 'max'
            save_best_only (bool): If True, only saves the model if the monitored metric improves
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def __call__(self, model: nn.Module, current_value: float, epoch: int):
        """
        Call this method after each validation epoch
        """
        should_save = False
        if self.save_best_only:
            if self.mode == 'min':
                if current_value < self.best_value:
                    self.best_value = current_value
                    should_save = True
            elif self.mode == 'max':
                if current_value > self.best_value:
                    self.best_value = current_value
                    should_save = True
        else:
            should_save = True
        
        if should_save:
            torch.save(model.state_dict(), self.filepath)
            logger.info(f"Epoch {epoch}: '{self.monitor}' improved to {current_value:.4f}. Model saved to {self.filepath}")
        else:
            logger.info(f"Epoch {epoch}: '{self.monitor}' did not improve. Model not saved.")
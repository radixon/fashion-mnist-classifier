import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json 
import os 
from typing import Tuple, List, Dict, Any 

class ModelEvaluator:
    """
    Manage the evaluation of a model
    """
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the ModelEvaluator

        Args:
            model (nn.Module): The model under evaluation
            device (torch.device): Perform the evaluation via the CPU or GPU [CUDA]
        """
        self.model = model.to(device) 
        self.device = device
        self.model.eval()   # Set model to evaluation mode

    def get_labels(self, dataloader: DataLoader) -> Tuple[List[int], List[int]]:
        """
        Collect predicted labels and true labels

        Args:
            dataloader (DataLoader): DataLoader for the dataset under evaluation
        
        Returns:
            Tuple[List[int], List[int]]:  A tuple containing lists of predicted values and true labels
        """
        predicted_labels = []
        true_labels = []

        with torch.no_grad():   # Disable gradient calculation for inference
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                yhat = self.model(inputs)
                _, predicted = torch.max(yhat.data, 1)  # Highest scoring class

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy()) 
        
        return true_labels, predicted_labels


    def evaluate_model_performance(self, true_labels: List[int], predicted_labels: List[int], class_names: List[str]) -> Dict[str, Any]:
        """
        Calculate performance metrics

        Args:
            true_labels (List[int]):  List of true labels
            predicted_labels (List[int]):  List of predicted labels
            class_names (List[str]): List of class names 
        
        Returns:
            Dict[str, Any]: Metrics
        """
        overall_accuracy = accuracy_score(true_labels, predicted_labels)

        # classification_report returns a str
        report_dict = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
        confusion_matrix_ = confusion_matrix(true_labels, predicted_labels)
        metrics = {"overall_accuracy": overall_accuracy, "classification_report": report_dict, "confusion_matrix": confusion_matrix_.tolist()}

        return metrics


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """
    Saves metrics to a JSON file

    Args:
        metrics (Dict[str, Any]): Dictionary of metrics to save
        save_path (str): Path to the JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {save_path}")



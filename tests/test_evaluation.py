import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__name__), '..')))

from evaluation.evaluator import ModelEvaluator
from models.cnn_models import VanillaCNN
from utils.helpers import FASHION_MNIST_CLASSES

class TestEvaluation:
    """
    Test suite for evaluation functionality.
    """
    @pytest.fixture
    def sample_model(self) -> VanillaCNN:
        """
        Create a model for testing.

        Returns:
            VanillaCNN: A Vanilla CNN model in evaluation mode
        """
        model = VanillaCNN(input_dim=(1, 28, 28), num_classes=10)
        model.eval()
        return model 
    
    @pytest.fixture
    def device(self) -> torch.device:
        """
        Get available device for testing.

        Returns:
            torch.device:  CPU device for testing
        """
        return torch.device("cpu")
    
    @pytest.fixture
    def sample_data(self) -> torch.utils.data.DataLoader:
        """
        Create sample data for testing.

        Returns:
            torch.utils.data.DataLoader: Sample data loader with test data
        """
        images = torch.randn(20, 1, 28, 28)

        # 10 Labels are required 
        labels = torch.tensor([i % 10 for i in range(20)])
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        return loader
    
    @pytest.fixture
    def evaluator_setup(self, sample_model: VanillaCNN, device: torch.device) -> ModelEvaluator:
        """
        Set up evaluator for testing.

        Args:
            sample_model (VanillaCNN): Model instance from fixture
            device (torch.device): Device from fixture

        Returns:
            ModelEvaluator:  Configured evaluator instance
        """
        return ModelEvaluator(sample_model, device)

    def test_evaluator_initialization(self, sample_model: VanillaCNN, device: torch.device):
        """
        Test ModelEvaluator initialization.
        """
        evaluator = ModelEvaluator(sample_model, device)
        assert evaluator.model is sample_model, "Evaluator should store model"
        assert evaluator.device is device, "Evaluator should store device"
    
    def test_get_predictions_and_labels_shapes(self, evaluator_setup: ModelEvaluator, sample_data: torch.utils.data.DataLoader):
        """
        Test get_labels returns the correct shapes.

        Args:
            evaluator_setup (ModelEvaluator): Evaluator instance from fixture
            sample_data (torch.utils.data.DataLoader): Sample data from fixture
        """
        y_true, y_pred = evaluator_setup.get_labels(sample_data)
        assert len(y_true) == 20, "There are 20 true labels"
        assert len(y_pred) == 20, "There are 20 prediction labels"
        assert isinstance(y_true, list), "True labels are present in a ndarray"
        assert isinstance(y_pred, list), "Predicted labels are present in a ndarray"
    
    def test_get_predictions_and_labels_values(self, evaluator_setup: ModelEvaluator, sample_data: torch.utils.data.DataLoader):
        """
        Verify prediction labels and values are in a valid range.

        Args:
            evaluator_setup (ModelEvaluator): Evaluator instance from fixture
            sample_data (torch.utils.data.DataLoader): Sample data from fixture
        """
        y_true, y_pred = evaluator_setup.get_labels(sample_data)

        # Labels valid range [0,9]
        assert np.all(np.array(y_true) >= 0 & (np.array(y_true) <= 9)), "True labels are in range [0, 9]"
        assert np.all(np.array(y_pred) >= 0 & (np.array(y_pred) <= 9)), "Predicted labels are in range [0, 9]"
    
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_evaluate_model_performance_structure(self, evaluator_setup: ModelEvaluator, sample_data: torch.utils.data.DataLoader):
        """
        Verify structure of evaluate_model_performance.

        Args:
            evaluator_setup (ModelEvaluator): Evaluator instance from fixture
            sample_data (torch.utils.data.DataLoader): Sample data from fixture
        """
        y_true, y_pred = evaluator_setup.get_labels(sample_data)

        metrics = evaluator_setup.evaluate_model_performance(y_true, y_pred, FASHION_MNIST_CLASSES)

        # Verify the existance of the required keys
        required_keys = ['overall_accuracy', 'classification_report_str', 'classification_report_dict', 'confusion_matrix']
        for key in required_keys:
            assert key in metrics, f"Verify {key} is a required metric"

        # Verify types
        overall_accuracy = metrics['overall_accuracy']
        report_str = metrics['classification_report_str']
        report_dict = metrics['classification_report_dict']
        confusion_matrix = metrics['confusion_matrix']

        assert isinstance(overall_accuracy, float), "Overall accuracy is of type: float"
        assert isinstance(report_dict, dict), "Classification report dict is of type: Dict"
        assert isinstance(report_str, str), "Classification report str is of type: str"
        assert isinstance(confusion_matrix, list), "Confusion matrix is of type: List"
    
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_evaluate_model_performance_accuracy_range(self, evaluator_setup: ModelEvaluator, sample_data: torch.utils.data.DataLoader):
        """
        Verify that accuracy is in a valid range.

        Args:
            evaluator_setup (ModelEvaluator): Evaluator instance from fixture
            sample_data (torch.utils.data.DataLoader): Sample data from fixture
        """
        y_true, y_pred = evaluator_setup.get_labels(sample_data)

        metrics = evaluator_setup.evaluate_model_performance(y_true, y_pred, FASHION_MNIST_CLASSES)

        accuracy = metrics['overall_accuracy']
        assert np.all((accuracy >= 0.0) and (accuracy <= 1.0)), "Accuracy is in the range [0.0, 1.0]"
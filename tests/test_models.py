import pytest
import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnn_models import VanillaCNN, DeepCNN
from models.resnet_model import FashionResNet

class TestModels:
    """
    Test suite for neural network model architectures.
    """
    @pytest.fixture
    def sample_input(self):
        """
        Create sample input data for testing.
        """
        return torch.randn(4, 1, 28, 28)
    
    @pytest.fixture
    def device(self):
        """
        Get available device for testing.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_vanilla_cnn_instantiation(self):
        """
        Test VanillaCNN instantiation.
        """
        model = VanillaCNN(input_dim=(1, 28, 28), num_classes=10)
        assert isinstance(model, nn.Module), "VanillaCNN should be a PyTorch Module"
        assert isinstance(model, VanillaCNN), "Model should be VanillaCNN instance"
    
    def test_vanilla_cnn_forward_pass(self, sample_input):
        """
        Test VanillaCNN forward pass.
        """
        model = VanillaCNN(input_dim=(1, 28, 28), num_classes=10)
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        
        # Verify output shape
        expected_shape = (4, 10)    # (batch_size, num_classes)
        assert output.shape == expected_shape, f"Expected Shape: {expected_shape}, Output Shape: {output.shape}"

        # Check output data type
        assert output.dtype == torch.float32, f"Excpected float32, Output dtype: {output.dtype}"

    def test_vanilla_cnn_output_range(self, sample_input):
        """
        Test the range of the VanillaCNN output.
        """
        model = VanillaCNN(input_dim=(1, 28, 28), num_classes=10)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)
        
        assert torch.all(torch.isfinite(output)), "All outputs should be finite"
        assert output.min() > -100 and output.max() < 100, "Logits should be in reasonable range [-100, 100]"
    
    def test_deep_cnn_instantiation(self):
        """
        Test DeepCNN instantiation.
        """
        model = DeepCNN(input_dim=(1, 28, 28), num_classes=10)
        assert isinstance(model, nn.Module), "DeepCNN should be a PyTorch Module"
        assert isinstance(model, DeepCNN), "Model should be DeepCNN instance"
    
    def test_deep_cnn_forward_pass(self, sample_input):
        """
        Test DeepCNN forward pass.
        """
        model = DeepCNN(input_dim=(1, 28, 28), num_classes=10)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)
        
        # Check output shape
        expected_shape = (4, 10)    # (batch_size, num_classes)
        assert output.shape == expected_shape, f"Expected Shape: {expected_shape}, Output Shape: {output.shape}"

        # Check output data type
        assert output.dtype == torch.float32, f"Expected float32, Output dtype: {output.dtype}"
    
    def test_deep_cnn_has_dropout(self):
        """
        Verify DeepCNN contains Dropout layers for regularization.
        """
        model = DeepCNN(input_dim=(1, 28, 28), num_classes=10)

        # Verify Dropout layers
        has_dropout = any(isinstance(module, nn.Dropout) for module in model.modules())
        assert has_dropout, "DeepCNN should contain Dropout layers for regularization"
    
    def test_deep_cnn_has_batchnorm(self):
        """
        Verify DeepCNN contains BatchNorm layers.
        """
        model = DeepCNN(input_dim=(1, 28, 28), num_classes=10)

        # Verify BatchNorm2d layers
        has_batchnorm = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
        assert has_batchnorm, "DeepCNN should contain BatchNorm2d layers"
    
    def test_fashion_resnet_instantiation(self):
        """
        Verify FashionResNet can be instantiated.
        """
        model = FashionResNet(input_dim=(1, 28, 28), num_classes=10)
        assert isinstance(model, nn.Module), "FashionResNet should be a PyTorch Module"
        assert isinstance(model, FashionResNet), "Model should be FashionResNet instance"
    
    def test_fashion_resnet_forward_pass(self, sample_input):
        """
        Verify FashionResNet forward pass.
        """
        model = FashionResNet(input_dim=(1, 28, 28), num_classes=10)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)
        
        # Verify shape
        expected_shape = (4, 10)    # (batch_size, num_classes)
        assert output.shape == expected_shape, f"Ecpected Shape: {expected_shape}, Output Shape: {output.shape}"

        # Verify output data type
        assert output.dtype == torch.float32, f"Expected float32, Output Data Type: {output.dtype}"

    def test_models_with_different_batch_sizes(self):
        """
        Test all models with different batch sizes.
        """
        models = [VanillaCNN(input_dim=(1,28,28), num_classes=10), DeepCNN(input_dim=(1,28,28), num_classes=10), 
                  FashionResNet(input_dim=(1,28,28), num_classes=10)]
        batch_sizes = [1, 16, 32, 64]

        for model in models:
            model.eval()
            for batch_size in batch_sizes:
                input_tensor = torch.randn(batch_size, 1, 28, 28)

                with torch.no_grad():
                    output = model(input_tensor)
                
                expected_shape = (batch_size, 10)
                assert output.shape == expected_shape, f"Model {type(model).__name__} failed with batch_size {batch_size}"
    
    def test_models_for_vanishing_gradient(self, sample_input):
        """
        Verify gradients flow through all models.
        """
        models = [VanillaCNN(input_dim=(1,28,28), num_classes=10), DeepCNN(input_dim=(1,28,28), num_classes=10),
                  FashionResNet(input_dim=(1,28,28), num_classes=10)]
        
        for model in models:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            # Forward Pass
            output = model(sample_input)

            # Dummy Targets
            targets = torch.randint(0, 10, (4,))

            # Backward Pass
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()

            # Verify gradients exists
            has_gradients = any(param.grad is not None and torch.sum(torch.abs(param.grad)) > 0 
                                for param in model.parameters())
            assert has_gradients, f"Model {type(model).__name__} should have gradients after backward pass"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_models_cuda_compatibility(self, sample_input):
        """
        Verify models work when CUDA is available.
        """
        device = torch.device("cuda")
        models = [VanillaCNN(input_dim=(1,28,28), num_classes=10), DeepCNN(input_dim=(1,28,28), num_classes=10),
                  FashionResNet(input_dim=(1,28,28), num_classes=10)]
        
        for model in models:
            model.to(device)
            model.eval()

            input_cuda = sample_input.to(device)

            with torch.no_grad():
                output = model(input_cuda)
            
            assert output.device.type == 'cuda', f"Output should be on CUDA for {type(model).__name__}"
            assert output.shape == (4, 10), f"Output shape should be {(4, 10)} on CUDA for {type(model).__name__}"
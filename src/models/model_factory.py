import os
import sys
import logging
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.cnn_models import VanillaCNN, DeepCNN
from src.models.resnet_model import FashionResNet

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating model instances based on configuration
    """
    # Available Models
    _MODELS = {
            "VanillaCNN": VanillaCNN,
            "DeepCNN": DeepCNN,
            "FashionResNet": FashionResNet
    }

    _MODEL_PARAM_KEYS = {
                    "VanillaCNN": [],
                    "DeepCNN": ["deep_cnn_params"],
                    "FashionResNet": ["fashion_resnet_params"]
    }

    @classmethod
    def create_model(cls,
                    model_name: str,
                    data_config: Dict[str, Any],
                    model_config: Dict[str, Any]) -> Any:
        """
        Create a model instance based on the model name and configuration.

        Args:
            model_name (str):   Name of the model to create
            data_config (Dict[str, Any]):   Data configuration containing input_shape and num_classes
            model_config (Dict[str, Any]):  Model configuration containing model-specific parameters

        Returns:
            Model instance of the requested type
        """
        if model_name not in cls._MODELS:
            available_models = list(cls._MODELS.keys())
            raise ValueError(f"Unknown model name: '{model_name}'. "
                             f"Available models: {available_models}")

        # Get model class
        model_class = cls._MODELS[model_name]

        # Prepare base parameters
        base_params = {
                    'input_dim': tuple(data_config['input_shape']),
                    'num_classes': data_config['num_classes']
        }

        # Get model-specific parameters
        model_specific_params = {}
        param_keys = cls._MODEL_PARAM_KEYS[model_name]

        for param_key in param_keys:
            if param_key in model_config:
                model_specific_params.update(model_config[param_key])
            else:
                logger.warning(f"Model parameter key '{param_key}' not found in config for {model_name}")

        # Combine parameters
        all_params = {**base_params, **model_specific_params}

        # Log model creation
        logger.info(f"Creating {model_name} model with parameters: {all_params}")

        # Create and return model instance
        try:
            model_instance = model_class(**all_params)
            logger.info(f"Successfully created {model_name} model")
            return model_instance
        except Exception as e:
            logger.error(f"Failed to create {model_name} model: {str(e)}")
            raise

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model names.

        Returns:
            List of available model names
        """
        return list(cls._MODELS)

    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """
        Check if a model name is valid.

        Args:
            model_name (str): Name of the model to validate

        Returns:
            bool: True if model name is valid, False otherwise
        """
        return model_name in cls._MODELS

    def create_model_from_config(model_name: str,
                    data_config: Dict[str, Any],
                    model_config: Dict[str, Any]) -> Any:
        """
        Convenience function to create a model instance.

        Args:
            model_name (str): Name of the model to create
            data_config (Dict[str, Any]): Data configuration
            model_config (Dict[str, Any]): Model configuration

        Returns:
            Model instance of the requested type
        """
        return ModelFactory.create_model(model_name, data_config, model_config)

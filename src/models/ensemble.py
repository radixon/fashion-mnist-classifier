import os
import sys
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.base_model import BaseModel
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)

class VotingEnsemble(nn.Module):
    """
    Ensemble model that combines predictions from multiple base models using soft voting.
    """
    
    def __init__(self, models: List[nn.Module], voting_type: str = "soft"):
        super(VotingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.voting_type = voting_type
        self.num_models = len(models)
        
        logger.info(f"Created VotingEnsemble with {self.num_models} models using {voting_type} voting")
    
    def forward(self, x):
        """Forward pass through ensemble."""
        if self.voting_type == "soft":
            # Soft voting: average the probabilities
            predictions = []
            for model in self.models:
                model_output = model(x)
                # Apply softmax to get probabilities
                model_probs = torch.softmax(model_output, dim=1)
                predictions.append(model_probs)
            
            # Average probabilities
            ensemble_probs = torch.stack(predictions).mean(dim=0)
            return ensemble_probs
        
        elif self.voting_type == "hard":
            # Hard voting: majority vote
            predictions = []
            for model in self.models:
                model_output = model(x)
                model_pred = torch.argmax(model_output, dim=1)
                predictions.append(model_pred)
            
            # Majority voting (simplified version)
            stacked_preds = torch.stack(predictions)
            ensemble_pred = torch.mode(stacked_preds, dim=0)[0]
            return ensemble_pred
        
        else:
            raise ValueError(f"Unsupported voting type: {self.voting_type}")

    @classmethod
    def create_ensemble(cls, 
                       model_names: List[str], 
                       data_config: Dict[str, Any], 
                       model_config: Dict[str, Any], 
                       model_weights_dir: str,
                       voting_type: str = "soft"):
        """
        Create an ensemble by loading multiple trained models using model factory.
        """
        models = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for model_name in model_names:
            try:
                # Create model using factory
                model_instance = ModelFactory.create_model(model_name, data_config, model_config)
                
                # Load weights
                model_path = f"{model_weights_dir}/best_model_{model_name}.pth"
                model_instance.load_state_dict(torch.load(model_path, map_location=device))
                model_instance.to(device)
                model_instance.eval()
                
                models.append(model_instance)
                logger.info(f"Successfully loaded {model_name} for ensemble")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name} for ensemble: {str(e)}")
                raise
        
        return cls(models, voting_type)
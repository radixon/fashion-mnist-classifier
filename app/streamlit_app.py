import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_models import VanillaCNN
from src.utils.config import load_config
from src.utils.helpers import FASHION_MNIST_CLASSES

@st.cache_resource
def load_model():
    """
    Load the trained model with caching.

    Returns:
        Optional[torch.nn.Module]: The loaded PyTorch model if successful
    """
    try:
        model_path = 'models/best_model.pth'
        if os.path.exists(model_path):
            model = VanillaCNN()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from typing import Tuple
import numpy.typing as npt

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

def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, npt.NDArray[np.float32]]:
    """
    This function converts the input image to grayscale, resizes it to 28x28 pixels, normalizes
    pixel values, and applies Fashion MNIST-specific transformations.

    Args:
        image (Image.Image): PIL Image object to preprocess
    
    Returns:
        Tuple[torch.Tensor, npt.NDArray[np.float32]]: A tuple containing
                    - Peprocessed model input (1, 1, 28, 28)
                    - Processed image array for visualization (28, 28)
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array and normalize
    image_array = np.array(image)

    # Invert colors if needed
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    
    # Normalize [0, 1]
    normalize_array = image_array.astype(np.float32)

    # Convert to tensor and add batch dimension
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    pil_image = Image.fromarray((normalize_array * 255).astype(np.uint8))
    tensor = transform(pil_image).unsqueeze(0)

    return tensor, normalize_array

def predict_image(model: torch.nn.Module, image_tensor: torch.Tensor) -> Tuple[int, float, npt.NDArray[np.float32]]:
    """
    Make predictions on processed images.

    Args:
        model (torch.nn.Module): The trained PyTorch model for inference
        image_tnesor (torch.Tensor): Preprocessed image tensor of shape (1, 1, 28, 28)
    
    Returns:
        Tuple[int, float, npt.NDArray[np.float32]]: A tuple containing
                    -   Predicted class index (0-9)
                    -   Confidence score for the predicted class (0.0 - 1.0)
                    -   Array of probabilities for all classes (length 10)
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        all_probabilities = probabilities[0].numpy()
    
    return predicted_class, confidence, all_probabilities

def display_prediction_results(predicted_class: int, confidence: float, all_probabilities: npt.NDArray[np.float32]) -> None:
    """
    Display prediction results in the Streamlit interface.

    Args:
        predicted_class (int): The predicted class index (0 - 9)
        confidence (float): Confidence score for the predicted class
        all_probabilities (npt.NDArray[np.float32]): Probabilities for all classes
    
    Returns:
        None
    """
    st.success(f"**Prediction:** {FASHION_MNIST_CLASSES[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2%}")

    # Show all probabilities
    st.subheader("All Class Probabilities")
    for i, (class_name, prob) in enumerate(zip(FASHION_MNIST_CLASSES, all_probabilities)):
        # Highlight the predicted class
        if i == predicted_class:
            st.write(f"**{class_name}: {prob:.2%}**")
        else:
            st.write(f"{class_name}: {prob:.2%}")

def create_probability_chart(all_probabilities: npt.NDArray[np.float32], predicted_class: int) -> plt.Figure:
    """
    Create a bar chart showing probability distribution across all classes.

    Args:
        all_probabilities (npt.NDArray[np.float32]):  Array of proabilities for all classes
        predicted_class (int): Index of the predicted class to highlight
    
    Returns:
        plt.Figure: Matplotlib figure object containing the probability chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(FASHION_MNIST_CLASSES)), all_probabilities)

    # Highlight the predicted class
    bars[predicted_class].set_color('red')

    ax.set_yticks(range(len(FASHION_MNIST_CLASSES)))
    ax.set_yticklabels(FASHION_MNIST_CLASSES)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities for All Classes')

    # Add Percentage Labels
    for i, prob in enumerate(all_probabilities):
        ax.text(prob + 0.01 i, f'{prob:.1%}', va='center')
    
    plt.tight_layout()
    return fig
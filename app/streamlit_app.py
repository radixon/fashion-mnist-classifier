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
from src.models.model_factory import ModelFactory
from src.models.ensemble import VotingEnsemble
from src.utils.config import load_config

FASHION_MNIST_CLASSES = [
                        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@st.cache_resource
def load_model():
    """
    Load the trained model with caching.

    Returns:
        Optional[torch.nn.Module]: The loaded PyTorch model if successful
    """
    config = load_config()
    data_config = config['data']
    model_config = config['model']
    paths_config = config['paths']
    training_config = config['training']
    device = torch.device("cpu")

    model = None
    model_name = "VanillaCNN"
    
    if model_name == "Ensemble":
        st.info("Loading Ensemble Model")
        if 'ensemble' not in config['model']:
            st.error("Ensemble configuration not found in config.yaml")
            st.stop()

        ensemble_config = config['model']['ensemble']
        model_names = ensemble_config['models_to_include']
        voting_type = ensemble_config['voting_type']

        try:
            model = VotingEnsemble.create_ensemble(
                model_names=model_names,
                data_config=data_config,
                model_config=model_config,
                model_weights_dir=paths_config['model_save_dir'],
                voting_type=voting_type
            )
            st.success(f"Ensemble loaded with base models: {model_names}")
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Ensemble model loading failed: {str(e)}")
            st.stop()
    try:
        model = ModelFactory.create_model_from_config(model_name, data_config, model_config)
        model_checkpoint_config = training_config['callbacks']['model_checkpoint']
        model_checkpoint_filepath = os.path.join(paths_config['model_save_dir'],
                                                 model_checkpoint_config['filepath'].format(model_name=model_name))
        
        model.load_state_dict(torch.load(model_checkpoint_filepath, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model creation failed: {str(e)}")
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
    config = load_config()
    model_name = config['model']['name']
    with torch.no_grad():
        outputs = model(image_tensor)
        if model_name == "Ensemble":
            probabilities = outputs[0]
        else:
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
        ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
    
    plt.tight_layout()
    return fig

def main() -> None:
    """
    Set up Streamlit interface, handles unser interactions, processes uploaded images,
    and displays prediction results.
    """
    st.set_page_config(
                        page_title="Fashion MNIST Classifier",
                        page_icon=":tshirt:",
                        layout="wide"
    )

    st.title(":shirt: Fashion MNIST Classifier")
    st.markdown("## Upload an image of clothing to classify the image!")

    # Sidebar Information
    st.sidebar.title("About")
    st.sidebar.info("A Convolutional Neural Network trained on the Fashion MNIST dataset "
                    "is used to classify clothing items into 10 categories.")
    
    st.sidebar.markdown("## Classes:")
    for i, class_name in enumerate(FASHION_MNIST_CLASSES):
        st.sidebar.write(f"{i}: {class_name}")
    
    # Load Model
    model = load_model()
    if model is None:
        st.error("Failed to load model.  Please train a model.")
        return

    # Create file uploader
    uploaded_file = st.file_uploader("Choose an image file",
                                     type=['png', 'jpg', 'jpeg'],
                                     help="Upload a grayscale or color image of clothing"
                                    )
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original Image", width=200)
            
            # Preprocess image
            image_tensor, processed_array = preprocess_image(image)

            with col1:
                st.image(processed_array,
                         caption="Processed Image (28x28)",
                         width=200,
                         clamp=True
                        )
            
            # Make Prediction
            predicted_class, confidence, all_probabilities = predict_image(model, image_tensor)

            with col2:
                display_prediction_results(predicted_class, confidence, all_probabilities)

                # Create and display probability chart
                fig = create_probability_chart(all_probabilities, predicted_class)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.exception(e) # Show traceback for debugging


if __name__ == "__main__":
    main()
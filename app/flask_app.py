import os
import sys
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_factory import ModelFactory
from src.models.ensemble import VotingEnsemble
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config('configs/config.yaml')
data_config = config['data']
training_config = config['training']
model_config = config['model']
paths_config = config['paths']
device = torch.device("cpu")

def load_model():
    """
    Load the trained model
    """
    model_name = model_config['name']
    try:
        # Create model instance
        if model_name == "Ensemble":
            logger.info("Loading Ensemble model")
            ensemble_config = config['ensemble']
            model_names = ensemble_config['models_to_include']
            voting_type = ensemble_config['voting_type']

            model = VotingEnsemble.create_ensemble(
                        model_names=model_names,
                        data_config=data_config,
                        model_config=model_config,
                        model_weights_dir=paths_config['model_save_dir'],
                        voting_type=voting_type
            )
            logger.info(f"Ensemble loaded with base models: {base_model_names}")
            return model
        else:
            logger.info(f"Loading {model_name} model")

            model = ModelFactory.create_model(model_name, data_config, model_config)

            # Load model weights
            model_checkpoint_config = training_config['callbacks']['model_checkpoint']
            full_model_load_path = os.path.join(
                                                paths_config['model_save_dir'],
                                                model_checkpoint_config['filepath'].format(model_name=model_name)
                                            )
            
            if os.path.exists(full_model_load_path):
                model.load_state_dict(torch.load(full_model_load_path, map_location='cpu'))
                logger.info(f"{model_name} model loaded successfully")
                model.to(device)
                model.eval()
                return model
            else:
                raise FileNotFoundError(f"Model '{model_name}' not found at {full_model_load_path}.")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    """
    status = "ok" if model is not None else "model_not_loaded"
    model_name = model_config['name'] if model_config is not None else "N/A"
    return jsonify({
        'status':   status,
        'model_loaded': model is not None
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Get all available classes
    """
    return jsonify({
                    'classes':  CLASS_NAMES,
                    'num_classes':  len(CLASS_NAMES)
    })

@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Prediction endpoint
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image from request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        image_tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            all_probabilities = probabilities[0].numpy().tolist()

        # Prepare response
        response = {
                    'predicted_class':  predicted_class,
                    'predicted_label':  CLASS_NAMES[predicted_class],
                    'confidence':   confidence,
                    'all_probabilities': {
                        CLASS_NAMES[i]: prob for i, prob in enumerate(all_probabilities)
                    }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error':    str(e)}), 500
    
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for prediction
    """
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image_array = np.array(image)

    # Invert colors if needed
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    
    # Normalize to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0

    # Convert to tensor
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))  
    ])

    pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
    tensor = transform(pil_image).unsqueeze(0)

    return tensor

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Fashion MNIST Classifier API',
        'endpoints': {
            '/health': 'GET - Health check',
            '/classes': 'GET - Available classes',
            '/predict': 'POST - Make prediction (requires base64 image in JSON)'
        }
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model_instance():
        print("Starting Flask API server")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model. Exiting!!!")
        sys.exit(1)
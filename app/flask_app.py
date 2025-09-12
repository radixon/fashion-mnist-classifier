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
from src.data.preprocessing import get_transforms
from src.models.model_factory import ModelFactory
from src.models.ensemble import VotingEnsemble
from src.utils.config import load_config
from src.utils.helpers import FASHION_MNIST_CLASSES

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
            if 'ensemble' not in config['model']:
                raise ValueError("Ensemble configuration not found in config.yaml")
            ensemble_config = config['model']['ensemble']
            model_names = ensemble_config['models_to_include']
            voting_type = ensemble_config['voting_type']

            model = VotingEnsemble.create_ensemble(
                        model_names=model_names,
                        data_config=data_config,
                        model_config=model_config,
                        model_weights_dir=paths_config['model_save_dir'],
                        voting_type=voting_type
            )
            logger.info(f"Ensemble loaded with base models: {model_names}")
            model.to(device)
            model.eval()
            return model
        else:
            logger.info(f"Loading {model_name} model")

            model = ModelFactory.create_model_from_config(model_name, data_config, model_config)

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

# Load model when app starts
try:
    model = load_model()
    logger.info(f"Model '{model_config['name']}' loaded successfully for Flask API")
except Exception as e:
    logger.error(f"Failed to load model on startup: {str(e)}")
    model = None

# Get transforms for inference
transform_for_inference = get_transforms(train=False)

# Flask App Setup
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint
    """
    status = "health" if model is not None else "unhealthy"
    model_name = model_config['name'] if model_config is not None else "N/A"
    return jsonify({
        'status':   status,
        'model_loaded': model is not None,
        'model_name': model_name,
        'message': f"Fashion MNIST Classifier API is running with {model_config['name']} model" if model is not None else "Model failed to load",
    }), 200 if model is not None else 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """
    Get all available classes
    """
    return jsonify({
                    'classes':  FASHION_MNIST_CLASSES,
                    'num_classes':  len(FASHION_MNIST_CLASSES)
    })

@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """
    Prediction endpoint
    """
    if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The model failed to load on startup'
                }), 500
    try:
        # Get image from request
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please provide base64 encoded image in "image" field'
                }), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({
                'error': 'Invalid image data',
                'message': f'Failed to decode base64 image: {str(e)}'
            }), 400

        # Preprocess image
        image_tensor = preprocess_image(image)
        model_name = model_config['name']
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            if model_name == "Ensemble":
                probabilities = outputs[0]
            else:
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            # Get Predicted Class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            all_probabilities = probabilities[0].numpy().tolist()

        # Prepare response
        response = {
                    'prediction':  predicted_class,
                    'predicted_label':  FASHION_MNIST_CLASSES[predicted_class],
                    'confidence':   confidence,
                    'all_probabilities': {
                        FASHION_MNIST_CLASSES[i]: prob for i, prob in enumerate(all_probabilities)
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
        'model': model_config['name'],
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/classes': 'GET - Available classes',
            '/predict': 'POST - Make prediction (requires base64 image in JSON)'
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors.
    """
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors.
    """
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occured'
    }), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask API server")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model. Exiting!!!")
        sys.exit(1)

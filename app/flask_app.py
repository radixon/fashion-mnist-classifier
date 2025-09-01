import os
import sys
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_models import VanillaCNN, DeepCNN
from src.models.resnet_model import FashionResNet
from src.utils.config import load_config

app = Flask(__name__)
CLASS_NAMES = [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

model = None
model_config = None
def load_model_instance():
    """
    Load the trained model
    """
    global model, model_config
    try:
        config = load_config('configs/config.yaml')
        data_config = config['data']
        model_config = config['model']
        training_config = config['training']
        paths_config = config['paths']
        model_name = model_config['name']

        # Create model instance
        if model_name == "VanillaCNN":
            model = VanillaCNN(
                input_dim=tuple(data_config['input_shape']),
                num_classes=data_config['num_classes']
            )
        elif model_name == "DeepCNN":
            model = DeepCNN(
                input_dim=tuple(data_config['input_shape']),
                num_classes=data_config['num_classes'],
                **model_config.get('deep_cnn_params', {})
            )
        elif model_name == "FashionResNet":
            model = FashionResNet(
                input_dim=tuple(data_config['input_shape']),
                num_classes=data_config['num_classes'],
                **model_config.get('fashion_resnet_params', {})
            )
        
        # Load model weights
        model_checkpoint_config = training_config['callbacks']['model_checkpoint']
        full_model_load_path = os.path.join(
                                            paths_config['model_save_dir'],
                                            model_checkpoint_config['filepath']
                                        )
        
        if os.path.exists(full_model_load_path):
            model.load_state_dict(torch.load(full_model_load_path, map_location='cpu'))
            model.eval()
            return True
        else:
            print(f"Model file not found: {full_model_load_path}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

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
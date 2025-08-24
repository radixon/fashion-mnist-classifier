# Welcome to the Fashion MNIST Classifier Project Documentation!

This documentation provides a comprehensive guide to the **Fashion MNIST Classifier**, a deep learning project built with PyTorch. This project showcases a complete machine learning pipeline, from data preparation and model development to training, evaluation, and experiment tracking.

## Project Goals

*   To implement a modular deep learning pipeline for image classification. 
*   To demonstrate best practices in MLOps, including configuration management, structured logging, and experiment tracking with MLflow. 
*   To explore various Convolutional Neural Network (CNN) architectures, from simple baselines to more advanced designs like ResNet. 
*   To provide a clear, reproducible, and well-documented example of a PyTorch project. 

## Key Features

### Architecture

-   Modular PyTorch implementation
-   Abstract base classes for extensibility
-   Type hints throughout codebase
-   Professional logging system

### Models Implemented

-   **VanillaCNN**: Baseline 2-layer CNN
-   **DeepCNN**: Enhanced VanillaCNN with BatchNorm and Dropout
-   **FashionResNet**: ResNet-like architecture with residual connections

### MLOps Integration

-   MLflow experiment tracking
-   Automated model logging and versioning
-   Performance metrics visualization
-   Hyperparameter tracking

### Quality Assurance

-   Comprehensive unit tests
-   Automated CI/CD with GitHub Actions
-   Code linting with flake8
-   Type checking support

## What You'll Find Here

*   **[Installation Guide](installation.md)**: How to set up your development environment.
*   **[Usage Instructions](usage.md)**: How to run the training, evaluation, and prediction scripts.
*   **[API Reference](api_reference.md)**: Detailed documentation for the project's Python modules and functions.

Start by [installing the project locally](installation.md) to get set up!
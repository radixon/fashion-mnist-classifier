# Installation Guide

This guide will walk you through setting up your development environment for the Fashion MNIST Classifier project.

## Prerequisites

Before you begin, ensure you have the following installed:

-   **Python 3.8 or higher**
-   **Git**
-   **pip** (Python package installer)

## Step-by-Step Installation

## 1. Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/radixon/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

## 2. Create and Activate a Python Virtual Environment

It's highly recommended to use a Python virtual environment to manage project dependencies and avoid conflicts with other Python projects or your system-wide Python installation.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # Linux
.venv\Scripts\activate  # Windows
```

## 3. Install Core Project Dependencies

Install the Python packages listed in `requirements.txt`.

```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

## 4. Verify Installation
```bash
# Run all tests
pytest tests/ -v
```

## 4. Install PyTorch and TorchVision

PyTorch installation varies based on your operating system and GPU (CUDA) availability. **You must install PyTorch and TorchVision separately using the command from their official website.**

1.  Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Select your desired configuration.
3.  Copy the provided installation command and run it in your activated virtual environment.

## 6. Download Data

The Fashion MNIST dataset will be automatically downloded the first instance of running training.

```bash
python3 scripts/train.py
```

Once these steps are complete, your development environment is ready!

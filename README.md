# Fashion MNIST Classifier: End-to-End Deep Learning Pipeline with MLOps


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

This project delivers a **comprehensive, end-to-end deep learning solution** for classifying images from the Fashion MNIST dataset. Designed with **PyTorch**, the project showcases a full machine learning lifecycle, from structured data management and model experimentation to evaluation and **integrated experiment tracking with MLflow**. The project emphasizes **MLOps** principles and best practices in software engineering.

## Table of Contents

1.  [Features](#features)
2.  [Key Results](#key-results)
3.  [Architecture Overview](#architecture-overview)
4.  [Dataset](#dataset)
5.  [Technologies](#technologies)
6.  [Installation](#installation)
7.  [Usage](#usage)
8.  [Project Structure](#project-structure)
9.  [MLOps & Experiment Tracking](#mlops--experiment-tracking)
10. [License](#license)

## Features

*   **Advanced Deep Learning Models:** Implements and experiments with various Convolutional Neural Network (CNN) architectures, including a `VanillaCNN` for baseline and a `DeepCNN` incorporating Batch Normalization and Dropout for improved performance and regularization.
*   **Efficient PyTorch Data Pipeline:** Leverages `torchvision.datasets` and `torch.utils.data.DataLoader` for streamlined data acquisition, preprocessing (normalization, tensor conversion), batching, and shuffling.
*   **Configurable Training & Evaluation:** Utilizes a centralized `config.yaml` for managing all hyperparameters, model architectures, and file paths, ensuring high reproducibility and easy experimentation.
*   **Robust Training Management:** Encapsulates the core PyTorch training and validation loops (`ModelTrainer`), handling crucial model state changes (`model.train()`, `model.eval()`).
*   **Comprehensive Evaluation & Visualization:** Provides tools for calculating key performance metrics (accuracy, precision, recall, F1-score) and generates insightful plots (confusion matrices, training history, sample predictions).
*   **Structured Logging:** Implements a sophisticated logging system that directs messages to both the console and timestamped log files, crucial for debugging, monitoring, and auditing training runs.
*   **Experiment Tracking with MLflow:** Fully integrates MLflow to automatically log parameters, epoch-wise metrics, and model artifacts for systematic experiment comparison and management.
*   **Exploratory Data Analysis (EDA):** Dedicated Jupyter Notebook for in-depth data understanding, visualization of sample images, and analysis of class distributions.
*   **Rapid Experimentation:** Jupyter Notebook for quick training runs and interactive analysis of baseline model performance.
*   **Modular Project Structure:** Organized into logical directories and Python packages (`src/`, `data/`, `configs/`, `scripts/`, `models/`, `results/`, `notebooks/`) for clear separation of concerns.
*   **Version Control:** Managed with Git and includes a `.gitignore` for a clean repository.

## Key Results

*   **Model Performance:** The `DeepCNN` model, leveraging Batch Normalization and Dropout, consistently achieves higher validation accuracy **~93%** after 10 epochs compared to the `VanillaCNN` **~91%**.
*   **Experiment Tracking:** Over **[Number of runs you have logged, e.g., 5-10]** unique experiment runs have been logged and can be compared using the MLflow UI, demonstrating efficient hyperparameter tuning and model selection.
*   **Reproducibility:** All training runs are fully reproducible, with parameters, metrics, and models versioned in MLflow.

## Architecture Overview

The project follows a modular and layered architecture:

```mermaid
graph TD
    A[Raw Data] --> B(Data Loading & Preprocessing)
    B --> C[Processed Data (Batches)]
    C -- Train --> D(Model Training)
    D -- Trained Model --> E[Saved Model Checkpoint]
    E -- Load --> F(Model Evaluation)
    F --> G[Metrics & Plots]
    G --> H[MLflow Tracking]
    H -- View --> I[MLflow UI]
    D -- Log --> H
    F -- Log --> H
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px
```


## Dataset

The project uses the Fashion MNIST dataset.  The dataset consists of 70,000 grayscale images categorized into 10 classes of fashion articles.

## Technologies

* Language: Python
* Deep Learning: PyTorch, torchvision
* Data & ML Libraries: NumPy, pandas, matplotlib, seaborn, tqdm, scikit-learn
* MLOps & Tools: MLflow, YAML, Git, logging

## Installation

### 1. Clone Repo

```bash
git clone https://github.com/radixon/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

### 2. [Optional] Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -e
pip install -r requirements.txt
```

## Usage
Explore the dataset, train, and evaluate the model using the configured scripts and notebooks:

- Train Model
```bash
python3 scripts/train.py
```

- Evaluate Model
```bash
python3 scripts/evaluate.py
```

### Jupyter Lab
- JupyterLab
```bash
jupyter lab
```
- 'notebooks/01_data_exploration.ipynb': Exploratory Data Analysis Notebook
- 'notebooks/02_baseline_models.ipynb': Baseline Model Experimentation Notebook

### MLflow UI
```bash
mlflow ui
```
- Navigate to the address displayed in terminal (http://<localhost>:5000)

## Project Structure
fashion-mnist-classifier/ <br/>
├── .git/                     # Git version control metadata<br/>
├── .venv/                    # Python virtual environment<br/>
├── configs/                  # Configuration files for the project<br/>
│   └── config.yaml           # Main configuration for data, training, model, and paths<br/>
├── data/                     # Stores all project data<br/>
│   ├── raw/                  # Original, raw dataset files (downloaded by torchvision)<br/>
│   │   └── FashionMNIST/     # (Contains the actual .gz data files)<br/>
│   │       └── raw/<br/>
│   │       └── processed/    # (Used by torchvision for some datasets, but empty for F-MNIST)<br/>
│   ├── processed/            # For cleaned/transformed data (currently empty)<br/>
│   └── external/             # For external/supplementary data (currently empty)<br/>
├── mlruns/                   # MLflow tracking data (ignored by Git)<br/>
├── notebooks/                # Jupyter Notebooks for exploration and experimentation<br/>
│   ├── 01_data_exploration.ipynb # In-depth EDA of the Fashion MNIST dataset<br/>
│   └── 02_baseline_models.ipynb  # Rapid experimentation with baseline models<br/>
├── scripts/                  # Executable Python scripts for various tasks<br/>
│   ├── train.py              # Main script to train the model<br/>
│   └── evaluate.py           # Script to evaluate a trained model<br/>
├── src/                      # Source code for the project<br/>
│   ├── __init__.py           # Marks 'src' as a Python package<br/>
│   ├── data/                 # Data loading and preprocessing modules<br/>
│   │   ├── __init__.py       # Marks 'src/data' as a Python sub-package<br/>
│   │   ├── data_loader.py    # Functions to load datasets and create DataLoaders<br/>
│   │   └── preprocessing.py  # Defines image transformations (ToTensor, Normalize)<br/>
│   ├── evaluation/           # Modules for model evaluation and visualization<br/>
│   │   ├── __init__.py       # Marks 'src/evaluation' as a Python sub-package<br/>
│   │   ├── evaluator.py      # Core logic for calculating metrics<br/>
│   │   └── visualization.py  # Functions for plotting results (e.g., confusion matrix)<br/>
│   ├── model/                # Neural network model definitions<br/>
│   │   ├── __init__.py       # Marks 'src/model' as a Python sub-package<br/>
│   │   ├── base_model.py     # Abstract base class for all models<br/>
│   │   └── cnn_models.py     # SimpleCNN and DeepCNN implementations<br/>
│   ├── training/             # Modules for training logic<br/>
│   │   ├── __init__.py       # Marks 'src/training' as a Python sub-package<br/>
│   │   └── trainer.py        # Manages the training and validation loops<br/>
│   │   └── callbacks.py      # (Placeholder, will be implemented soon)<br/>
│   │   └── metrics.py        # (Placeholder, will be implemented soon)<br/>
│   └── utils/                # General utility functions<br/>
│       ├── __init__.py       # Marks 'src/utils' as a Python sub-package<br/>
│       ├── config.py         # Handles loading configuration from YAML<br/>
│       ├── helpers.py        # General helper functions (e.g., timestamp, class names)<br/>
│       └── logger.py         # Centralized logging setup<br/>
├── .gitignore                # Specifies files/folders to be ignored by Git<br/>
├── LICENSE                   # Project's license (e.g., MIT)<br/>
├── README.md                 # Project overview and documentation<br/>
├── requirements.txt          # Python dependencies<br/>
└── setup.py                  # Project packaging configuration<br/>


## MLOps & Experiment Tracking

- Configuration Management: All parameters are in config.yaml for reproducibility
- Structured Logging: logs generated to aid in debugging
- Experiment Tracking: MLflow integrated to record and compare each experiment run

## License
This project is licensed under the [MIT License](LICENSE).

# Fashion MNIST Classifier

An end-to-end deep learning pipeline for classifying Fashion MNIST images.
<br/>
<br/>

## Project Features

* Convolutional Neural Networks
* Config-driven training via YAML
* Modular codebase
* Clean documentation
<br/>
<br/>

## Tech Stack

* Python 3.8+
* PyTorch
* NumPy, pandas, scikit-learn
* Matplotlib, seaborn
* PyYAML

## Getting Started

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

### 1. Exploratory Data Analysis
    * Start JupyterLab from project root:
    ```bash
    jupyter lab
    ```
    * Open 'notebooks/01_data_exploration.ipynb'.
    * Run cells to visualize sample images, analyze class distributions, and save sample image plots and class distribution charts.

### 2. Train the Model
    ```bash
    python3 scripts/train.py
    ```
    * Loads configuration, train VanillaCNN model, and save state dictionary.

### 3. Evaluate the Trained Model
    ```bash
    python3 scripts/evaluate.py
    ```
    *  Load the saved model, evaluate the model on the test set, save detailed metrics, and generate a confusion matrix.
    * Log file for the evaluation created and saved in 'relusts/logs/'.

## License
This project is licensed under the [MIT License](LICENSE).

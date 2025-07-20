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

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Training Script
```bash
python3 scripts/train.py
```

## Notes

* The script includes logic to detect CUDA if available

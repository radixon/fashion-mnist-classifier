# Usage Instructions

This section details how to use the various scripts and notebooks provided in this project.

Ensure you have completed the [Installation Guide](installation.md) and your virtual environment is activated before proceeding.

## 1. Configuration

All key parameters for data, training, model, and paths are managed in `configs/config.yaml`. Before running any training or evaluation, review this file to adjust settings as needed.

## 2. Running the Scripts

All main scripts are located in the `scripts/` directory and should be executed from the project's root directory.

### Train the Model

The `train.py` script orchestrates the model training process. It loads configurations, prepares data, initializes the chosen model, and runs the training loop. The 'train.py' script also logs experiment details to MLflow and saves the best model checkpoint.

```bash
python3 scripts/train.py
```

*   **Output:** Training progress will be displayed in the console and logged to a timestamped file in `results/logs/`. MLflow will record parameters, metrics, and artifacts in the `mlruns/` directory. The best model will be saved to `models/best_model.pth`.

### Evaluate the Model

The `evaluate.py` script loads a trained model, evaluates its performance on the test set, and generates detailed reports and visualizations.

```bash
python3 scripts/evaluate.py
```

*   **Output:** Evaluation results will be displayed in the console and logged. Detailed metrics (JSON) and a confusion matrix plot will be saved to `results/metrics/` and `results/figures/` respectively.

### Make Predictions (Coming Soon)

The `predict.py` script will allow you to make predictions on new data using a saved model.

```bash
python3 scripts/predict.py --image_path path/to/your/image.png
```

## 3. Using Jupyter Notebooks

Jupyter Notebooks are used for exploratory data analysis (EDA) and experimentation.

### Start JupyterLab

From the project's root directory, launch JupyterLab:

```bash
jupyter lab
```

This will open JupyterLab in your web browser. Navigate to the `notebooks/` directory.

### Explore the Dataset (EDA)

Open `01_data_exploration.ipynb`. Run all cells to:

*   Visualize sample Fashion MNIST images.
*   Analyze class distributions.
*   Understand the dataset's characteristics.

Plots generated during EDA will be saved to `results/figures/`.

### Baseline Model Experimentation

Open `02_baseline_models.ipynb`. Run all cells to:

*   Train the `VanillaCNN` model interactively.
*   Observe training and validation loss/accuracy curves.
*   Generate and save plots of training history and confusion matrices.
*   Experiment with quick parameter changes.

## 4. MLflow UI

To view and compare your experiment runs, start the MLflow UI:

```bash
mlflow ui
```

Then, open your web browser and navigate to the address displayed in your terminal (usually `http://localhost:5000`).

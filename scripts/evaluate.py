import os
import sys
import torch
import numpy as np
from datetime import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import load_datasets, get_dataloaders
from src.data.preprocessing import get_transforms
from src.models.model_factory import ModelFactory
from src.models.ensemble import VotingEnsemble
from src.evaluation.evaluator import ModelEvaluator, save_metrics, save_classification_report_txt
from src.evaluation.visualization import plot_confusion_matrix, plot_sample_predictions
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.utils.helpers import get_timestamp_str, FASHION_MNIST_CLASSES

logger = logging.getLogger(__name__)

def main():
    print("----- Fashion MNIST Evaluation Script -----")

    # Load Configuration
    config = load_config()
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    paths_config = config['paths']

    timestamp = get_timestamp_str()
    log_filename = paths_config['eval_log_filename'].format(timestamp=timestamp)
    setup_logging(paths_config['logs_dir'], log_filename)
    logger.info("----- Fashion MNIST Evaluation Script (Logging Enabled) -----")

    # Device Configuration
    device = torch.device("cuda" if training_config['device'] == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    # Data Loading
    logger.info("\nLoading Test Dataset")
    transform = get_transforms(train=False) # Use Validation Transforms
    test_dataset_raw, test_dataset = load_datasets(raw_data_path=data_config['raw_data_path'], transforms=transform)
    _, test_loader = get_dataloaders(train_dataset=None, test_dataset=test_dataset, batch_size=training_config['batch_size'], 
                                 num_workers=training_config['num_workers'], pin_memory=training_config['pin_memory'])  

    # Model Loading
    model_name = model_config['name']
    model = None
    if model_name == "Ensemble":
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
        logger.info("Loaded Ensemble model for evaluation")
    else:
        try:
            model = ModelFactory.create_model_from_config(model_name, data_config, model_config)

            model_checkpoint_config = training_config['callbacks']['model_checkpoint']
            model_checkpoint_filepath = os.path.join(paths_config['model_save_dir'], model_checkpoint_config['filepath'].format(model_name=model_name))

            # Load the Best Model
            if os.path.exists(model_checkpoint_filepath):
                model.load_state_dict(torch.load(model_checkpoint_filepath, map_location=device))
                logger.info(f"Loaded {model_name} model for evaluation")
            else:
                logger.error(f"Checkpoint not found at {model_checkpoint_filepath}")
                sys.exit(1)
        except ValueError as e:
            logger.error(f"Model creation failed: {str(e)}") 
            sys.exit(1)
    
    logger.info("Model Successfully Loaded")

    # Model Evaluation
    logger.info("\nStarting Model Evaluation")
    evaluator = ModelEvaluator(model=model, device=device)
    true_labels, predicted_labels = evaluator.get_labels(test_loader)

    # Calculate Metrics
    metrics = evaluator.evaluate_model_performance(true_labels=true_labels, predicted_labels=predicted_labels, class_names=FASHION_MNIST_CLASSES)

    # Save Metrics as JSON & text file
    # JSON
    timestape = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"{paths_config['evaluation_metrics_filename_prefix']}{model_name}_{timestamp}.json"
    metrics_save_path = os.path.join(paths_config['metrics_save_dir'], metrics_filename)
    save_metrics(metrics=metrics, save_path=metrics_save_path)

    # text flie
    report_txt_filename = f"{paths_config['classification_report_filename_prefix']}{model_name}_{timestamp}.txt"
    report_txt_save_path = os.path.join(paths_config['metrics_save_dir'], report_txt_filename)
    save_classification_report_txt(metrics["classification_report_str"], report_txt_save_path)

    # Visualize Results
    conf_matrix = metrics["confusion_matrix"]
    conf_matrix_filename = f"{paths_config['confusion_matrix_filename_prefix']}{model_name}_{timestamp}.png"
    conf_matrix_save_path = os.path.join(paths_config['figures_save_dir'], conf_matrix_filename)
    plot_confusion_matrix(cm=np.array(conf_matrix), class_names=FASHION_MNIST_CLASSES, save_path=conf_matrix_save_path, normalize=True, 
                            title=f'Confusion Matrix ({model_name} Accuracy: {metrics["overall_accuracy"]:.4f})')
    
    # Plot Sample Predictions
    logger.info("\nGenerating Sample Predictions Plot")
    sample_preds_filename = f"sample_predicitons_{model_name}_{timestamp}.png"
    sample_preds_save_path = os.path.join(paths_config['figures_save_dir'], sample_preds_filename)

    # Batch images from test_loader to plot
    sample_images_batch, _ = next(iter(test_loader))

    # Raw images for ploting
    num_samples_to_plot = 25

    # Get first N raw images and image labels
    raw_sample_images = torch.stack([test_dataset_raw[i][0] for i in range(num_samples_to_plot)])
    raw_sample_true_labels = [test_dataset_raw[i][1] for i in range(num_samples_to_plot)]

    # Get predictions for selected raw images
    transform_for_inference = get_transforms(train=False)

    if isinstance(test_dataset_raw[0][0], torch.Tensor):
        transformed_raw_images = torch.stack([test_dataset_raw[i][0] for i in range(num_samples_to_plot)])
    else:
        transformed_raw_images = torch.stack([transform_for_inference(test_dataset_raw[i][0]) for i in range(num_samples_to_plot)])

    # Make Predictions
    model.eval()
    with torch.no_grad():
        outputs = model(transformed_raw_images.to(device))
        _, sample_predicted_labels = torch.max(outputs.data, 1)
    sample_predicted_labels = sample_predicted_labels.cpu().numpy().tolist()

    # Plot Sample Predictions
    plot_sample_predictions(raw_sample_images, raw_sample_true_labels, sample_predicted_labels, FASHION_MNIST_CLASSES, sample_preds_save_path,
                            num_samples=num_samples_to_plot, title=f'Sample Predictions ({model_name})')
    
    logger.info("\n===== Evaluation Complete =====")
    logger.info(f"Accuracy: {metrics['overall_accuracy']:.4f}")
    logger.info(f"\nClassification Report (saved to {report_txt_save_path}):\n")
    logger.info(metrics["classification_report_str"])
    

if __name__ == "__main__":
    try:
        import numpy as np
        import sklearn
        import torch
    except ImportError:
        print("Error: Libraries are missing. Install libraries: pip install numpy scikit-learn torch")
        sys.exit(1)
    main()
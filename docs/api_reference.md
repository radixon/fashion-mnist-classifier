```markdown
# API Reference

This section provides an auto-generated API reference for the core modules in the `src/` directory. The documentation is generated directly from the docstrings in the Python source code.

## DataLoader Module
::: src.data.data_loader
    options:
        members:
            - load_fashion_mnist_datasets
            - get_data_loaders

## Preprocessing Module
::: src.data.preprocessing
    options:
        members:
            - get_fashion_mnist_transforms

## Base Model Module
::: src.model.base_model
    options:
        members:
            - BaseModel

## CNN Module
::: src.model.cnn_models
    options:
        members:
            - VanillaCNN
            - DeepCNN

## ResNet-like Model Module
::: src.model.resnet_model
    options:
        members:
            - FashionResNet
            - BasicBlock

## Trainer Module
::: src.training.trainer
    options:
        members:
            - ModelTrainer

## Callbacks Module
::: src.training.callbacks
    options:
        members:
            - EarlyStopping
            - ModelCheckpoint

## Metrics Module
::: src.training.metrics
    options:
        members:
            - calculate_f1_score

## Evaluator Module
::: src.evaluation.evaluator
    options:
        members:
            - ModelEvaluator
            - save_metrics
            - save_classification_report_txt

## Visualization Module
::: src.evaluation.visualization
    options:
        members:
            - plot_confusion_matrix
            - plot_training_history
            - plot_sample_predictions

## Config Module
::: src.utils.config
    options:
        members:
            - load_config

## Helpers Module
::: src.utils.helpers
    options:
        members:
            - get_timestamp_str
            - FASHION_MNIST_CLASSES
```
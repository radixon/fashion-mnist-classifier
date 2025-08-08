import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from typing import List, Dict

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, normalize: bool=False, title: str='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix

    Args:
        cm (np.ndarray): sklearn.metrics.confusion_matrix
        class_names (List[str]): class labels
        save_path (str): Path to save the plot
        normalize (bool): Normalizes the confusion matrix
        title (str): Title of the plot
        cmap: Colormap for the plot
    """
    if normalize:
        cmp = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix without normalization')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Labels")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix saved to: {save_path}")

def plot_training_history(history: Dict[str, List[float]], save_path: str, title: str = 'Training History'):
    """
    Plots training and validation loss and accuracy

    Args:
        history (Dict[str, List[float]]): Dictionary containing matrics
        save_path (str): Path to save the plot
        title (str): Title of the plot
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    if 'train_accuracy' in history:
        plt.plot(history['train_accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to: {save_path}")
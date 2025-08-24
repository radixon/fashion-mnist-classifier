import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from typing import List, Dict


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          save_path: str,
                          normalize: bool = False,
                          title: str = 'Confusion Matrix',
                          cmap=plt.cm.Blues) -> None:
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
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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


def plot_training_history(history: Dict[str, List[float]],
                          save_path: str,
                          title: str = 'Training History') -> None:
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


def plot_sample_predictions(images: torch.Tensor,
                            true_labels: List[int],
                            predicted_labels: List[int],
                            class_names: List[str],
                            save_path: str,
                            num_samples: int = 25,
                            title: str = 'Sample Predictions') -> None:
    """
    Plots a grid of sample images with their true and predicted labels.

    Args:
        images (torch.Tensor): A batch of images tensors
        true_labels (List[int]): List of true labels corresponding to the images
        predicted_labels (List[int]): List of predicted labels
        class_names (List[str]): List of class names
        save_path (str): Full path to save the plot
        num_samples (int): Number of samples to display in the grid
        title (str): Title of plot
    """
    # Plot 25 images max
    num_samples = min(num_samples, len(images), 25)

    # Determine Grid Size
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    # Create Figure and Subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Flatten the axes
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]

        # Denormalize assuming original normalization was (x - 0.5) / 0.5
        display_image = (images[i].cpu().numpy() * 0.5 + 0.5).clip(0, 1)

        # PyTorch images (channel, height, width).  Maplotlib grayscale should be (height, width).
        ax.imshow(display_image.squeeze(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # Set Color and Title
        isCorrect = (true_labels[i] == predicted_labels[i])
        color = "green" if isCorrect else "red"
        ax.set_title(f"True: {class_names[true_labels[i]]} \nPredicted: {class_names[predicted_labels[i]]}",
                     color=color, fontsize=10)

    # Hide unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save Figures
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Sample predictions plot saved to: {save_path}")

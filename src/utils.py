import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import json

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None):
    """
    Plot and save a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, save_path=None):
    """
    Plot training history for neural network.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_model_results(results, save_path):
    """
    Save model results to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    results_copy = results.copy()
    for model_name in results_copy:
        if 'confusion_matrix' in results_copy[model_name]:
            results_copy[model_name]['confusion_matrix'] = \
                results_copy[model_name]['confusion_matrix'].tolist()
    
    with open(save_path, 'w') as f:
        json.dump(results_copy, f, indent=4)

def load_model_results(load_path):
    """
    Load model results from a JSON file.
    """
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    for model_name in results:
        if 'confusion_matrix' in results[model_name]:
            results[model_name]['confusion_matrix'] = \
                np.array(results[model_name]['confusion_matrix'])
    
    return results 
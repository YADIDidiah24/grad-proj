import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, jaccard_score, roc_curve, auc
)
from preprocessing import get_test_data

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        float: Dice coefficient.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if len(y_pred.shape) == 5 and len(y_true.shape) == 4:
        y_true = tf.expand_dims(y_true, axis=-1)

    axes = list(range(1, len(y_true.shape)))  # [1,2,3] or [1,2,3,4]
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_loss(y_true, y_pred):
    """
    Calculate Dice loss.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        float: Dice loss.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Calculate combined binary cross-entropy and Dice loss.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        float: Combined loss.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def evaluate_model(model, X_test, y_test=None, test_generator=None):
    """
    Evaluate model performance with multiple metrics.

    Args:
        model: Trained model.
        X_test: Test images.
        y_test: Test labels.
        test_generator: Test data generator (optional).

    Returns:
        tuple: (metrics_dict, X_test, y_test, y_pred, y_pred_prob)
    """
    try:
        if test_generator is not None and (X_test is None or y_test is None):
            try:
                X_test, y_test = get_test_data(test_generator)
            except Exception as e:
                print(f"Error getting test data from generator: {str(e)}")
                return {}, None, None, None, None

        # Check and adjust input shape for prediction if needed
        if X_test is not None:
            input_shape = model.input_shape
            X_test_shape = X_test.shape

            # If model expects 5D input but we have 4D data
            if len(input_shape) == 5 and len(X_test_shape) == 4:
                X_test = np.expand_dims(X_test, axis=-1)
                print(f"Adjusted X_test shape from {X_test_shape} to {X_test.shape}")

            # If model expects 4D input but we have 5D data
            elif len(input_shape) == 4 and len(X_test_shape) == 5:
                # This would be unusual, but handle it just in case
                X_test = X_test.reshape(-1, *X_test_shape[2:])
                print(f"Adjusted X_test shape from {X_test_shape} to {X_test.shape}")

        # Make predictions
        y_pred_prob = model.predict(X_test)

        # Ensure predictions and labels have same shape for evaluation
        if y_pred_prob.shape != y_test.shape:
            print(f"Shape mismatch - y_pred_prob: {y_pred_prob.shape}, y_test: {y_test.shape}")
            # Reshape if needed
            if len(y_pred_prob.shape) > len(y_test.shape):
                # Remove extra dimensions
                y_pred_prob = y_pred_prob.reshape(y_test.shape)
            elif len(y_pred_prob.shape) < len(y_test.shape):
                # Add extra dimensions
                for _ in range(len(y_test.shape) - len(y_pred_prob.shape)):
                    y_pred_prob = np.expand_dims(y_pred_prob, axis=-1)

        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Ensure we're working with proper dimensions for evaluation
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        y_pred_prob_flat = y_pred_prob.flatten()

        def dice_score(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            total = y_true.sum() + y_pred.sum()
            return 2 * intersection / total if total != 0 else float("nan")

        metrics = {
            "Accuracy": accuracy_score(y_test_flat, y_pred_flat),
            "F1 Score": f1_score(y_test_flat, y_pred_flat),
            "Precision": precision_score(y_test_flat, y_pred_flat),
            "Recall": recall_score(y_test_flat, y_pred_flat),
            "AUC-ROC": roc_auc_score(y_test_flat, y_pred_prob_flat),
            "Log Loss": log_loss(y_test_flat, y_pred_prob_flat),
            "MCC": matthews_corrcoef(y_test_flat, y_pred_flat),
            "Balanced Accuracy": balanced_accuracy_score(y_test_flat, y_pred_flat),
            "IoU": jaccard_score(y_test_flat, y_pred_flat, average="macro") if len(set(y_test_flat)) > 1 else float("nan"),
            "Sensitivity": recall_score(y_test_flat, y_pred_flat, pos_label=1),
            "Dice Score": dice_score(y_test_flat, y_pred_flat)
        }

        if len(set(y_test_flat)) > 1:
            try:
                tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_flat).ravel()
                metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            except ValueError:
                print("Could not compute confusion matrix, possibly due to class imbalance")
                metrics["Specificity"] = float("nan")
        else:
            metrics["Specificity"] = float("nan")  # Undefined if only one class is present

        return metrics, X_test, y_test, y_pred, y_pred_prob

    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        return {}, X_test, y_test, None, None

def visualize_slices(image, true_label, predicted_label, slice_index=None, model_name="Model"):
    """
    Visualize slices of 3D medical images with their true and predicted labels.
    
    Args:
        image: 3D image volume.
        true_label: True segmentation mask.
        predicted_label: Predicted segmentation mask.
        slice_index: Index of the slice to visualize (None for automatic selection).
        model_name: Name of the model for plot title.
    """
    # Print input shapes for debugging
    print(f"Input shapes - Image: {image.shape}, True Label: {true_label.shape}, Predicted Label: {predicted_label.shape}")
    
    # Handle batch dimension if present
    if len(image.shape) > 3:
        if image.shape[0] == 1:  # Batch size of 1
            image = image[0]  # Remove batch dimension
        else:
            image = image[0]  # Take first sample if batch > 1
            print("Warning: Taking only the first sample from batch")
    
    if len(true_label.shape) > 3:
        if true_label.shape[0] == 1:
            true_label = true_label[0]
        else:
            true_label = true_label[0]
    
    if len(predicted_label.shape) > 3:
        if predicted_label.shape[0] == 1:
            predicted_label = predicted_label[0]
        else:
            predicted_label = predicted_label[0]
    
    # Handle channel dimension if present (should now be 3D or 3D+channel)
    if len(image.shape) == 4 and image.shape[3] == 1:
        image = image[:, :, :, 0]
    elif len(image.shape) == 4 and image.shape[3] > 1:
        image = image[:, :, :, 0]  # Take first channel if multi-channel
        print("Warning: Taking only the first channel from multi-channel image")
    
    if len(true_label.shape) == 4 and true_label.shape[3] == 1:
        true_label = true_label[:, :, :, 0]
    elif len(true_label.shape) == 4 and true_label.shape[3] > 1:
        true_label = true_label[:, :, :, 0]
    
    if len(predicted_label.shape) == 4 and predicted_label.shape[3] == 1:
        predicted_label = predicted_label[:, :, :, 0]
    elif len(predicted_label.shape) == 4 and predicted_label.shape[3] > 1:
        predicted_label = predicted_label[:, :, :, 0]
    
    # Print shapes after processing
    print(f"Processed shapes - Image: {image.shape}, True Label: {true_label.shape}, Predicted Label: {predicted_label.shape}")
    
    # Now all should be 3D arrays
    if len(image.shape) != 3 or len(true_label.shape) != 3 or len(predicted_label.shape) != 3:
        print(f"Error: Expected 3D arrays after processing, got Image: {image.shape}, True Label: {true_label.shape}, Predicted Label: {predicted_label.shape}")
        # Try to flatten to 3D if possible
        if len(image.shape) > 3:
            image = image.reshape(image.shape[:3])
        if len(true_label.shape) > 3:
            true_label = true_label.reshape(true_label.shape[:3])
        if len(predicted_label.shape) > 3:
            predicted_label = predicted_label.reshape(predicted_label.shape[:3])
    
    # For 3D volumes, select a central slice if none specified
    if slice_index is None:
        # Find a slice with the most segmentation
        slice_index = np.argmax(np.sum(true_label, axis=(1, 2)))
    
    # Extract 2D slices
    image_slice = image[slice_index, :, :]
    true_label_slice = true_label[slice_index, :, :]
    predicted_label_slice = predicted_label[slice_index, :, :]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show raw image slice
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Input Image Slice")
    axes[0].axis("off")
    
    # Show true label slice overlay
    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(true_label_slice, cmap="jet", alpha=0.5)
    axes[1].set_title("True Label Overlay")
    axes[1].axis("off")
    
    # Show predicted label slice overlay
    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(predicted_label_slice, cmap="jet", alpha=0.5)
    axes[2].set_title("Predicted Label Overlay")
    axes[2].axis("off")
    
    plt.suptitle(f"{model_name} - Slice {slice_index}")
    plt.tight_layout()
    return fig


import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def compare_models(model_results=None, csv_file='model_comparison.csv', metrics_to_plot=None):
    """
    Compare multiple models using bar charts.

    Args:
        model_results: List of dictionaries containing model metrics.
        csv_file: Path to the CSV file containing model metrics.
        metrics_to_plot: List of metrics to include in the comparison.

    Returns:
        DataFrame: Comparison data.
        Figure: Comparison plot.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["Dice Score", "Accuracy", "F1 Score", "Precision", "Recall"]

    # Try to read the comparison DataFrame from the CSV file
    if model_results is None:
        try:
            df = pd.read_csv(csv_file, index_col='Model')
        except FileNotFoundError:
            raise ValueError(f"CSV file '{csv_file}' not found.")
    else:
        df = pd.DataFrame(model_results)
        df = df.set_index("Model")

    # Select only the metrics we want to plot
    df_plot = df[metrics_to_plot]

    # Plot the metrics
    fig, ax = plt.subplots(figsize=(15, 8))
    df_plot.plot(kind="bar", ax=ax)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)

    plt.tight_layout()

    return df, fig

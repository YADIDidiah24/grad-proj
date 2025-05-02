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

        y_pred = (y_pred_prob > 0.3).astype(int)  # Convert probabilities to binary predictions

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

import matplotlib.pyplot as plt
import numpy as np



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


def visualize_3d_volume(image, true_label, predicted_label, threshold=0.5, model_name="Model"):
    """
    Create interactive 3D visualizations of medical image volumes with true and predicted labels.
    
    Args:
        image: 3D image volume (D, H, W).
        true_label: True segmentation mask (D, H, W).
        predicted_label: Predicted segmentation mask (D, H, W).
        threshold: Threshold for binary visualization of probability maps.
        model_name: Name of the model for plot title.
        
    Returns:
        tuple: Three plotly figure objects (image volume, true label, predicted label)
    """
    # Ensure we're working with properly formatted 3D arrays
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
    
    if len(true_label.shape) == 4 and true_label.shape[3] == 1:
        true_label = true_label[:, :, :, 0]
    elif len(true_label.shape) == 4 and true_label.shape[3] > 1:
        true_label = true_label[:, :, :, 0]
    
    if len(predicted_label.shape) == 4 and predicted_label.shape[3] == 1:
        predicted_label = predicted_label[:, :, :, 0]
    elif len(predicted_label.shape) == 4 and predicted_label.shape[3] > 1:
        predicted_label = predicted_label[:, :, :, 0]
    
    # Ensure binary labels for visualization
    if np.max(true_label) <= 1.0 and np.min(true_label) >= 0.0 and not np.array_equal(true_label, true_label.astype(bool)):
        # Probably a probability map
        true_label = true_label > threshold
    
    if np.max(predicted_label) <= 1.0 and np.min(predicted_label) >= 0.0 and not np.array_equal(predicted_label, predicted_label.astype(bool)):
        # Probably a probability map
        predicted_label = predicted_label > threshold
    
    # Create 3D volume rendering using Plotly
    import plotly.graph_objects as go
    from skimage import measure
    
    # Create 3D surface for the image volume (isosurface)
    # Normalize the image to 0-1 range for better visualization
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Create a figure for the image volume
    fig_volume = go.Figure()
    
    # Try different iso-values to find a good representation
    iso_values = [0.3, 0.5, 0.7]  # Can be adjusted based on data distribution
    for iso_val in iso_values:
        try:
            # Extract isosurface vertices and faces
            verts, faces, _, _ = measure.marching_cubes(normalized_image, level=iso_val)
            
            # Create a mesh3d trace
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            
            fig_volume.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.3,
                color='lightgray',
                name=f'Image (iso={iso_val})'
            ))
            
            # If we successfully created one isosurface, we can break
            break
        except:
            continue
    
    # Create a figure for the true label
    fig_true = go.Figure()
    
    # Extract isosurface from true label (binary)
    try:
        verts, faces, _, _ = measure.marching_cubes(true_label.astype(float), level=0.5)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        fig_true.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.6,
            color='green',
            name='True Label'
        ))
    except:
        # If marching_cubes fails (e.g., no surface found)
        print("Warning: Could not extract surface from true label")
    
    # Create a figure for the predicted label
    fig_pred = go.Figure()
    
    # Extract isosurface from predicted label (binary)
    try:
        verts, faces, _, _ = measure.marching_cubes(predicted_label.astype(float), level=0.5)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        fig_pred.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.6,
            color='red',
            name='Predicted Label'
        ))
    except:
        # If marching_cubes fails (e.g., no surface found)
        print("Warning: Could not extract surface from predicted label")
    
    # Update layout for all figures
    for fig, title in zip([fig_volume, fig_true, fig_pred], 
                         [f"{model_name} - Volume", 
                          f"{model_name} - True Label", 
                          f"{model_name} - Predicted Label"]):
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
    
    return fig_volume, fig_true, fig_pred


def visualize_3d_comparison(image, true_label, predicted_label, threshold=0.5, model_name="Model"):
    """
    Create a combined interactive 3D visualization showing image volume with 
    true and predicted label overlays.
    
    Args:
        image: 3D image volume (D, H, W).
        true_label: True segmentation mask (D, H, W).
        predicted_label: Predicted segmentation mask (D, H, W).
        threshold: Threshold for binary visualization of probability maps.
        model_name: Name of the model for plot title.
        
    Returns:
        plotly figure object: Combined 3D visualization
    """
    # Ensure we're working with properly formatted 3D arrays (same as in visualize_3d_volume)
    if len(image.shape) > 3:
        if image.shape[0] == 1:
            image = image[0]
        else:
            image = image[0]
    
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
    
    # Handle channel dimension
    if len(image.shape) == 4 and image.shape[3] == 1:
        image = image[:, :, :, 0]
    elif len(image.shape) == 4 and image.shape[3] > 1:
        image = image[:, :, :, 0]
    
    if len(true_label.shape) == 4 and true_label.shape[3] == 1:
        true_label = true_label[:, :, :, 0]
    elif len(true_label.shape) == 4 and true_label.shape[3] > 1:
        true_label = true_label[:, :, :, 0]
    
    if len(predicted_label.shape) == 4 and predicted_label.shape[3] == 1:
        predicted_label = predicted_label[:, :, :, 0]
    elif len(predicted_label.shape) == 4 and predicted_label.shape[3] > 1:
        predicted_label = predicted_label[:, :, :, 0]
    
    # Ensure binary labels
    if np.max(true_label) <= 1.0 and np.min(true_label) >= 0.0 and not np.array_equal(true_label, true_label.astype(bool)):
        true_label = true_label > threshold
    
    if np.max(predicted_label) <= 1.0 and np.min(predicted_label) >= 0.0 and not np.array_equal(predicted_label, predicted_label.astype(bool)):
        predicted_label = predicted_label > threshold
    
    import plotly.graph_objects as go
    from skimage import measure
    
    # Create a combined figure
    fig = go.Figure()
    
    # Add image volume isosurface
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    iso_values = [0.3, 0.5, 0.7]
    volume_added = False
    
    for iso_val in iso_values:
        try:
            verts, faces, _, _ = measure.marching_cubes(normalized_image, level=iso_val)
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.2,
                color='lightgray',
                name=f'Image Volume'
            ))
            volume_added = True
            break
        except:
            continue
    
    # Add true label surface
    try:
        verts, faces, _, _ = measure.marching_cubes(true_label.astype(float), level=0.5)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.7,
            color='green',
            name='True Label'
        ))
    except:
        print("Warning: Could not extract surface from true label")
    
    # Add predicted label surface
    try:
        verts, faces, _, _ = measure.marching_cubes(predicted_label.astype(float), level=0.5)
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.7,
            color='red',
            name='Predicted Label'
        ))
    except:
        print("Warning: Could not extract surface from predicted label")
    
    # Update layout
    fig.update_layout(
        title=f"3D Comparison",
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Add buttons to toggle visibility of each component
    buttons = [
        dict(
            args=[{"visible": [True, True, True]}],
            label="Show All",
            method="update"
        ),
        dict(
            args=[{"visible": [True, False, False]}],
            label="Image Only",
            method="update"
        ),
        dict(
            args=[{"visible": [False, True, False]}],
            label="True Label Only",
            method="update"
        ),
        dict(
            args=[{"visible": [False, False, True]}],
            label="Predicted Label Only",
            method="update"
        ),
        dict(
            args=[{"visible": [True, True, False]}],
            label="Image + True Label",
            method="update"
        ),
        dict(
            args=[{"visible": [True, False, True]}],
            label="Image + Predicted Label",
            method="update"
        ),
        dict(
            args=[{"visible": [False, True, True]}],
            label="True + Predicted Labels",
            method="update"
        )
    ]
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    
    return fig
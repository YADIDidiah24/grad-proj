import os
import time
import psutil
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import pandas as pd
import matplotlib.pyplot as plt
import keras
from models import (
    build_3d_unet,
    build_3d_vnet,
    build_sa_net,
    build_e1d3_unet,
    build_3d_hdc_net,
)

from metrics import (     evaluate_model,
    visualize_slices,
    compare_models)
TARGET_SHAPE = (64, 64, 64)
BATCH_SIZE = 2

def resize_3d_image(image, target_shape, order=0):
    """
    Resizes a 3D image to the target shape.

    Args:
        image (ndarray): Input 3D image.
        target_shape (tuple): Target shape (depth, height, width).
        order (int): Interpolation order (0=nearest, 1=linear, etc).

    Returns:
        ndarray: Resized image.
    """
    if len(image.shape) == 2:
        print(f"Warning: Found 2D image with shape {image.shape}, expanding to 3D")
        # Add a third dimension (depth of 1)
        image = np.expand_dims(image, axis=0)

    if len(image.shape) != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")

    # Calculate zoom factors
    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=order)

def preprocess_data(images, labels):
    """
    Preprocesses the images and labels.

    Args:
        images (ndarray): Input images.
        labels (ndarray): Input labels.

    Returns:
        tuple: Preprocessed images and labels.
    """
    if len(images.shape) == 2 or (len(images.shape) == 3 and images.shape[-1] != 1):
        print(f"Warning: Reshaping single image from {images.shape}")
        images = np.expand_dims(images, axis=0)  # Add batch dimension

    if len(labels.shape) == 2 or (len(labels.shape) == 3 and labels.shape[-1] != 1):
        print(f"Warning: Reshaping single label from {labels.shape}")
        labels = np.expand_dims(labels, axis=0)  # Add batch dimension

    if np.max(images) > 0:
        images = images / np.max(images)

    labels = (labels > 0).astype(np.uint8)

    if len(images.shape) == 3:
        images = images[..., np.newaxis]
    if len(labels.shape) == 3:
        labels = labels[..., np.newaxis]

    print(f"Preprocessed shapes - Images: {images.shape}, Labels: {labels.shape}")
    return images, labels

def split_data(images, labels, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        images (ndarray): Input images.
        labels (ndarray): Input labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        tuple: Training and testing sets.
    """
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)

class NiftiDataGenerator(Sequence):
    """
    Data generator for loading and preprocessing NIfTI files in batches.
    """
    def __init__(self, image_dir, label_dir, image_filenames, label_filenames, batch_size=2, target_shape=(64, 64, 64)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = image_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.epoch = 0

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for img_file, lbl_file in zip(batch_images, batch_labels):
            img_data = nib.load(os.path.join(self.image_dir, img_file)).get_fdata()
            lbl_data = nib.load(os.path.join(self.label_dir, lbl_file)).get_fdata()

            img_resized = resize_3d_image(img_data, self.target_shape, order=1)  # Bilinear for images
            lbl_resized = resize_3d_image(lbl_data, self.target_shape, order=0)  # Nearest-neighbor for labels

            img_resized = img_resized / np.max(img_resized) if np.max(img_resized) > 0 else img_resized
            lbl_resized = (lbl_resized > 0).astype(np.uint8)

            # Add channel dimension
            images.append(img_resized[..., np.newaxis])
            labels.append(lbl_resized[..., np.newaxis])

        # Ensure shape is [batch_size, h, w, d, 1]
        images_array = np.array(images)
        labels_array = np.array(labels)

        return images_array, labels_array


def load_nifti_files(image_dir, label_dir, target_shape):
    """
    Loads NIfTI files from the specified directories.

    Args:
        image_dir (str): Path to the image directory.
        label_dir (str): Path to the label directory.
        target_shape (tuple): Target shape for resizing.

    Returns:
        tuple: Loaded images and labels.
    """
    images, labels = [], []
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    print(f"Found {len(image_files)} image files and {len(label_files)} label files")

    if len(image_files) != len(label_files):
        print("Warning: Number of image and label files don't match")
        num_files = min(len(image_files), len(label_files))
        image_files = image_files[:num_files]
        label_files = label_files[:num_files]

    for img_file, lbl_file in zip(image_files, label_files):
        try:
            img_path = os.path.join(image_dir, img_file)
            lbl_path = os.path.join(label_dir, lbl_file)

            img_data = nib.load(img_path).get_fdata()
            lbl_data = nib.load(lbl_path).get_fdata()

            if len(img_data.shape) == 2:
                img_data = np.expand_dims(img_data, axis=0)
            if len(lbl_data.shape) == 2:
                lbl_data = np.expand_dims(lbl_data, axis=0)

            img_data = resize_3d_image(img_data, target_shape, order=1)
            lbl_data = resize_3d_image(lbl_data, target_shape, order=0)
            images.append(img_data)
            labels.append(lbl_data)
        except Exception as e:
            continue

    if not images:
        raise ValueError("No valid image/label pairs were loaded")

    return np.array(images), np.array(labels)

def get_test_data(generator):
    """
    Gets test data from the generator.

    Args:
        generator (NiftiDataGenerator): Data generator.

    Returns:
        tuple: Test data and labels.
    """
    X_test = []
    y_test = []
    for i in range(len(generator)):
        x, y = generator[i]
        X_test.append(x)
        y_test.append(y)

    return np.concatenate(X_test), np.concatenate(y_test)

def train_model(model_type, data_params, training_params, model_params=None, progress_callback=None):
    """
    Trains a specified model.

    Args:
        model_type (str): Type of model to train.
        data_params (dict): Data parameters.
        training_params (dict): Training parameters.
        model_params (dict): Model-specific parameters.
        progress_callback (callable): Callback function to update progress.

    Returns:
        dict: Training results.
    """
    if model_params is None:
        model_params = {}

    start_time = time.time()
    process = psutil.Process()

    if model_type.lower() == 'unet':
        target_shape = model_params.get('target_shape', TARGET_SHAPE)
        use_generator = model_params.get('use_generator', True)
    elif model_type.lower() == 'vnet':
        target_shape = model_params.get('target_shape', TARGET_SHAPE)
        use_generator = model_params.get('use_generator', True)
    elif model_type.lower() == 'sanet':
        target_shape = model_params.get('target_shape', TARGET_SHAPE)
        use_generator = model_params.get('use_generator', True)
    elif model_type.lower() == 'e1d3_unet':
        target_shape = model_params.get('target_shape', TARGET_SHAPE)
        use_generator = model_params.get('use_generator', True)
    elif model_type.lower() == 'hdcnet':
        target_shape = model_params.get('target_shape', TARGET_SHAPE)
        use_generator = model_params.get('use_generator', True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if use_generator:
        image_files = sorted(os.listdir(data_params['image_path']))
        label_files = sorted(os.listdir(data_params['label_path']))
        train_img, test_img, train_lbl, test_lbl = train_test_split(
            image_files, label_files,
            test_size=data_params.get('test_size', 0.2),
            random_state=data_params.get('random_state', 42)
        )
        batch_size = training_params.get('batch_size', BATCH_SIZE)
        train_generator = NiftiDataGenerator(
            data_params['image_path'], data_params['label_path'],
            train_img, train_lbl, batch_size, target_shape
        )
        test_generator = NiftiDataGenerator(
            data_params['image_path'], data_params['label_path'],
            test_img, test_lbl, batch_size, target_shape
        )
        X_test, y_test = None, None
    else:
        print(f"Loading data with target shape {target_shape}...")
        images, labels = load_nifti_files(
            data_params['image_path'], data_params['label_path'], target_shape
        )
        print("Preprocessing data...")
        images, labels = preprocess_data(images, labels)
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = split_data(
            images, labels, test_size=data_params.get('test_size', 0.2)
        )
        train_generator, test_generator = None, None

    print(f"Building {model_type} model...")
    input_shape = target_shape + (1,)
    use_dice_loss = model_params.get('use_dice_loss', False)

    if model_type.lower() == 'unet':
        model = build_3d_unet(input_shape, use_dice_loss)
    elif model_type.lower() == 'vnet':
        model = build_3d_vnet(input_shape, use_dice_loss)
    elif model_type.lower() == 'sanet':
        model = build_sa_net(input_shape, use_dice_loss)
    elif model_type.lower() == 'e1d3_unet':
        num_filters = model_params.get('num_filters', 16)
        model = build_e1d3_unet(input_shape, use_dice_loss, num_filters)
    elif model_type.lower() == 'hdcnet':
        model = build_3d_hdc_net(input_shape, use_dice_loss)

    print(f"Starting training for {model_type}...")
    epochs = training_params.get('epochs', 20)

    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            if progress_callback:
                progress_callback(progress)

    progress_callback_instance = ProgressCallback()

    if use_generator:
        history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[progress_callback_instance])
    else:
        batch_size = training_params.get('batch_size', BATCH_SIZE)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[progress_callback_instance])

    end_time = time.time()
    training_time = end_time - start_time
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    print(f"Evaluating {model_type} model...")
    metrics, X_test_eval, y_test_eval, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test, test_generator)

    metrics["Training Time (s)"] = training_time
    metrics["Memory (MB)"] = memory_usage

    return {
        "model_name": model_type,
        "model": model,
        "history": history,
        "metrics": metrics,
        "test_data": (X_test_eval, y_test_eval, y_pred, y_pred_prob)
    }


def run_segmentation_pipeline(models_to_train, data_paths, save_models=True, compare=True, visualize=True, training_params=None, specific_model_params=None, progress_callback=None):
    """
    Runs the segmentation pipeline for training multiple models.

    Args:
        models_to_train (list): List of models to train.
        data_paths (dict): Data paths.
        save_models (bool): Whether to save the trained models.
        compare (bool): Whether to compare the models.
        visualize (bool): Whether to visualize the results.
        training_params (dict): Training parameters.
        specific_model_params (dict): Model-specific parameters.
        progress_callback (callable): Callback function to update progress.

    Returns:
        tuple: Comparison DataFrame and trained models.
    """
    if training_params is None:
        training_params = {
            'epochs': 2,
            'batch_size': BATCH_SIZE
        }

    if specific_model_params is None:
        specific_model_params = {}

    model_results = []
    all_trained_models = {}

    for model_type in models_to_train:
        print(f"\n{'='*50}\nTraining {model_type}\n{'='*50}")
        model_params = specific_model_params.get(model_type, {})

        try:
            result = train_model(model_type, data_paths, training_params, model_params, progress_callback)
            metrics = result['metrics']
            metrics['Model'] = model_type.upper()
            model_results.append(metrics)
            all_trained_models[model_type] = result

            if visualize and result['test_data'][0] is not None:
                X_test, y_test, y_pred, y_pred_prob = result['test_data']

                if y_test is not None and y_pred is not None and y_pred_prob is not None:
                    y_test_flat = y_test.flatten()
                    y_pred_flat = y_pred.flatten()
                    y_pred_prob_flat = y_pred_prob.flatten()

                    visualize_slices(y_test_flat, y_pred_flat, y_pred_prob_flat, model_name=model_type.upper())

                    try:
                        visualize_slices(
                            X_test[0] if len(X_test.shape) > 3 else X_test,
                            y_test[0] if len(y_test.shape) > 3 else y_test,
                            y_pred[0] if len(y_pred.shape) > 3 else y_pred,
                            model_name=model_type.upper()
                        )
                    except Exception as e:
                        print(f"Error visualizing slices: {e}")
                        print(f"Shapes - X_test: {X_test.shape}, y_test: {y_test.shape}, y_pred: {y_pred.shape}")

            if save_models:
                model_filename = f"{model_type}_model.h5"
                result['model'].save(model_filename)
                print(f"Model saved to {model_filename}")

        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            print("Skipping this model and continuing with others.")
            continue

    print("Trained models:", all_trained_models.keys())  # Debugging statement

    if compare and len(model_results) > 1:
        comparison_df = compare_models(model_results)
        print("Type of comparison_df:", type(comparison_df))  # Debugging statement
        print("Content of comparison_df:", comparison_df)  # Debugging statement

        if isinstance(comparison_df, tuple):
            comparison_df = comparison_df[0]  # Assuming the DataFrame is the first element of the tuple

        comparison_df.to_csv("model_comparison.csv")
        print("Comparison saved to model_comparison.csv")
    else:
        comparison_df = pd.DataFrame(model_results).set_index('Model') if model_results else pd.DataFrame()
        if model_results:
            comparison_df.to_csv(f"{models_to_train[0]}_results.csv")

    return comparison_df, all_trained_models

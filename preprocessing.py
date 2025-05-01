import numpy as np
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
from utils import resize_3d_image

def preprocess_data(images, labels):
    """
    Preprocess images and labels for model training, ensuring consistent shapes.

    Args:
        images (ndarray): Input images.
        labels (ndarray): Input labels.

    Returns:
        tuple: (preprocessed_images, preprocessed_labels)
    """
    # Handle single image case
    if len(images.shape) == 2 or (len(images.shape) == 3 and images.shape[-1] != 1):
        print(f"Warning: Reshaping single image from {images.shape}")
        images = np.expand_dims(images, axis=0)  # Add batch dimension
    if len(labels.shape) == 2 or (len(labels.shape) == 3 and labels.shape[-1] != 1):
        print(f"Warning: Reshaping single label from {labels.shape}")
        labels = np.expand_dims(labels, axis=0)  # Add batch dimension

    # Normalize images
    if np.max(images) > 0:
        images = images / np.max(images)

    # Binarize labels
    labels = (labels > 0).astype(np.uint8)

    # Add channel dimension if needed
    if len(images.shape) == 3:
        images = images[..., np.newaxis]
    if len(labels.shape) == 3:
        labels = labels[..., np.newaxis]

    # Ensure 4D shape: [batch, height, width, depth, channel] format required by some models
    if len(images.shape) == 4 and len(images.shape[1:]) == 3:  # If [batch, h, w, d]
        images = np.expand_dims(images, axis=-1)  # Add channel: [batch, h, w, d, 1]
    if len(labels.shape) == 4 and len(labels.shape[1:]) == 3:  # If [batch, h, w, d]
        labels = np.expand_dims(labels, axis=-1)  # Add channel: [batch, h, w, d, 1]

    print(f"Preprocessed shapes - Images: {images.shape}, Labels: {labels.shape}")
    return images, labels


def split_data(images, labels, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        images (ndarray): Input images.
        labels (ndarray): Input labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)

def load_nifti_files(image_dir, label_dir, target_shape):
    """
    Load NIfTI files from directories and resize them.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): Directory containing label files.
        target_shape (tuple): Target shape for resizing.
        
    Returns:
        tuple: (images, labels) as numpy arrays
    """
    images, labels = [], []

    # Get sorted lists of files
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    print(f"Found {len(image_files)} image files and {len(label_files)} label files")

    # Make sure we have matching pairs
    if len(image_files) != len(label_files):
        print("Warning: Number of image and label files don't match")
        # Use the smaller number
        num_files = min(len(image_files), len(label_files))
        image_files = image_files[:num_files]
        label_files = label_files[:num_files]

    for img_file, lbl_file in zip(image_files, label_files):
        try:
            img_path = os.path.join(image_dir, img_file)
            lbl_path = os.path.join(label_dir, lbl_file)

            img_data = nib.load(img_path).get_fdata()
            lbl_data = nib.load(lbl_path).get_fdata()

            # Handle 2D data
            if len(img_data.shape) == 2:
                img_data = np.expand_dims(img_data, axis=0)
            if len(lbl_data.shape) == 2:
                lbl_data = np.expand_dims(lbl_data, axis=0)

            # Resize images and labels
            img_data = resize_3d_image(img_data, target_shape, order=1)  # Bilinear for images
            lbl_data = resize_3d_image(lbl_data, target_shape, order=0)  # Nearest-neighbor for labels
            
            images.append(img_data)
            labels.append(lbl_data)
        except Exception as e:
            print(f"Error loading {img_file} or {lbl_file}: {str(e)}")
            continue

    if not images:
        raise ValueError("No valid image/label pairs were loaded")

    return np.array(images), np.array(labels)

def get_test_data(generator):
    """
    Extract all test data from a generator.
    
    Args:
        generator: Data generator.
        
    Returns:
        tuple: (X_test, y_test) concatenated from all batches
    """
    X_test = []
    y_test = []
    for i in range(len(generator)):
        x, y = generator[i]
        X_test.append(x)
        y_test.append(y)

    return np.concatenate(X_test), np.concatenate(y_test)
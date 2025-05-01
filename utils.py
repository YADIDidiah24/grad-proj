import os
import re
import zipfile
import subprocess
import numpy as np
import nibabel as nib
import shutil
import requests
from tqdm import tqdm
from skimage.io import imread
from scipy.ndimage import zoom
import streamlit as st
import random
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def download_and_extract_kaggle_dataset(kaggle_dataset_url):
    """
    Downloads and extracts a Kaggle dataset using the Kaggle API with progress bar.
    
    Args:
        kaggle_dataset_url (str): URL of the Kaggle dataset.
        
    Returns:
        str: Path to the extracted dataset directory.
    """
    match = re.search(r'kaggle\.com/datasets/([^/]+/[^/]+)', kaggle_dataset_url)
    if not match:
        raise ValueError("Invalid Kaggle dataset URL")
    dataset_id = match.group(1)

    dataset_name = dataset_id.split('/')[-1]
    zip_file = f"{dataset_name}.zip"
    extract_dir = dataset_name

    # Check if dataset is already extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        st.success(f"Dataset already extracted in '{extract_dir}/'. Skipping download.")
        return extract_dir

    # Check if ZIP is already downloaded
    if not os.path.exists(zip_file):
        with st.spinner(f"Downloading dataset: {dataset_id}"):
            # Using subprocess with stdout capture to monitor download progress
            process = subprocess.Popen(
                ['kaggle', 'datasets', 'download', '-d', dataset_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Create a progress bar in Streamlit
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Monitor the download progress
            for line in process.stdout:
                if "%" in line:
                    try:
                        # Extract percentage from Kaggle output
                        percent = int(re.search(r'(\d+)%', line).group(1))
                        progress_bar.progress(percent / 100)
                        progress_text.text(f"Downloading: {percent}% complete")
                    except (AttributeError, ValueError):
                        pass
                        
            process.wait()
            
            if process.returncode != 0:
                raise Exception("Failed to download dataset")
                
            progress_bar.progress(1.0)
            progress_text.text("Download complete!")
            st.success(f"Successfully downloaded {zip_file}")
    else:
        st.info(f"Zip file '{zip_file}' already exists. Skipping download.")

    # Extract the ZIP with progress
    os.makedirs(extract_dir, exist_ok=True)
    with st.spinner(f"Extracting to {extract_dir}/"):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get total files for progress tracking
            total_files = len(zip_ref.namelist())
            
            # Create a progress bar for extraction
            extract_progress = st.progress(0)
            extract_text = st.empty()
            
            # Extract with progress updates
            for i, file in enumerate(zip_ref.namelist()):
                zip_ref.extract(file, extract_dir)
                extract_progress.progress((i + 1) / total_files)
                if (i + 1) % max(1, int(total_files / 20)) == 0:  # Update text less frequently
                    extract_text.text(f"Extracting: {i + 1}/{total_files} files")
                    
            extract_text.text(f"Extracted {total_files} files")
            st.success("Extraction complete!")

    return extract_dir

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

def convert_tiff_to_nifti(tiff_path, output_path):
    """
    Convert a TIFF stack to NIfTI format.
    
    Args:
        tiff_path (str): Path to the TIFF file.
        output_path (str): Path to save the NIfTI file.
        
    Returns:
        str: Path to the created NIfTI file.
    """
    data = imread(tiff_path)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)
    print(f"Converted {tiff_path} to {output_path}")
    return output_path

def move_first_set_files(base_dir):
    """
    Move files from the first set folder to root folders.
    
    Args:
        base_dir (str): Base directory containing set folders.
    """
    set_dirs = sorted([d for d in os.listdir(base_dir) if d.lower().startswith("set-")])
    if not set_dirs:
        print("No set folders found.")
        return

    first_set = set_dirs[0]
    set_path = os.path.join(base_dir, first_set)

    images_path = os.path.join(set_path, "images")
    masks_path = os.path.join(set_path, "masks")

    # Create output folders if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    # Move image files
    if os.path.exists(images_path):
        for file in os.listdir(images_path):
            if file.endswith((".tif", ".tiff")):
                src = os.path.join(images_path, file)
                dst = os.path.join("images", file)
                shutil.copy2(src, dst)

    # Move mask/label files
    if os.path.exists(masks_path):
        for file in os.listdir(masks_path):
            if file.endswith((".tif", ".tiff")):
                src = os.path.join(masks_path, file)
                dst = os.path.join("labels", file)
                shutil.copy2(src, dst)

    print(f"Moved files from {first_set}/images and {first_set}/masks to root folders.")

def is_image_folder(folder_name):
    """Check if a folder name indicates it contains images."""
    return any(keyword in folder_name.lower() for keyword in ['image', 'images', 'img', 'imgs'])

def is_label_folder(folder_name):
    """Check if a folder name indicates it contains labels or masks."""
    return any(keyword in folder_name.lower() for keyword in ['label', 'labels', 'mask', 'masks', 'annot'])

def find_image_and_label_folders(base_dir):
    """
    Find image and label folders in the dataset directory.
    
    Args:
        base_dir (str): Base directory to search.
        
    Returns:
        tuple: (image_folder_path, label_folder_path)
    """
    # Create output folders
    os.makedirs("images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)
    
    image_folder, label_folder = None, None

    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            if is_image_folder(d) and image_folder is None:
                image_folder = full_path
            elif is_label_folder(d) and label_folder is None:
                label_folder = full_path

        for file in files:
            if file.endswith(".tif") or file.endswith(".tiff"):
                file_path = os.path.join(root, file)
                fname = os.path.splitext(file)[0].lower()

                if "groundtruth" in fname or "label" in fname or "mask" in fname:
                    output_path = os.path.join("labels", f"{fname}.nii.gz")
                    convert_tiff_to_nifti(file_path, output_path)
                    label_folder = "labels"
                else:
                    output_path = os.path.join("images", f"{fname}.nii.gz")
                    convert_tiff_to_nifti(file_path, output_path)
                    image_folder = "images"

        if image_folder and label_folder:
            break

    return image_folder, label_folder

def display_statistics(image_data, label_data):
    img_min, img_max = np.min(image_data), np.max(image_data)
    img_mean, img_std = np.mean(image_data), np.std(image_data)
    foreground_voxels = np.sum(label_data > 0)
    total_voxels = label_data.size
    foreground_percentage = (foreground_voxels / total_voxels) * 100
    label_unique = np.unique(label_data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Image Min", f"{img_min:.2f}")
    col2.metric("Image Max", f"{img_max:.2f}")
    col3.metric("Image Mean", f"{img_mean:.2f}")
    col4.metric("Image StdDev", f"{img_std:.2f}")
    st.metric("Foreground Percentage", f"{foreground_percentage:.2f}%")
    st.write(f"**Label Values:** {label_unique}")

def visualize_slice(image_data, label_data, axis, slice_idx, view_mode, cmap):
    if axis == 0:
        img_slice = image_data[slice_idx, :, :]
        label_slice = label_data[slice_idx, :, :]
    elif axis == 1:
        img_slice = image_data[:, slice_idx, :]
        label_slice = label_data[:, slice_idx, :]
    else:
        img_slice = image_data[:, :, slice_idx]
        label_slice = label_data[:, :, slice_idx]

    img_norm = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)

    fig, ax = plt.subplots(figsize=(3, 3)) 
    if view_mode == "Image Only":
        ax.imshow(img_norm, cmap='gray')
    elif view_mode == "Label Only":
        ax.imshow(label_slice, cmap='hot')
    else:
        ax.imshow(img_norm, cmap='gray')
        ax.imshow(label_slice, cmap=cmap, alpha=0.5)
    ax.axis('off')

    # Show figure in smaller column to reduce visual space
    col1, _ = st.columns([1, 3])
    with col1:
        st.pyplot(fig)

def get_matched_pairs(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith(('.nii', '.nii.gz'))]
    matched = []
    for img in image_files:
        base = os.path.splitext(img)[0].replace('_image', '').replace('image_', '')
        label = next((l for l in label_files if base in l or base == l.replace('_label', '').replace('label_', '')), None)
        if label:
            matched.append((img, label))
    return matched

def explore_dataset(image_dir, label_dir, num_samples=1):
    matched_pairs = get_matched_pairs(image_dir, label_dir)
    if not matched_pairs:
        st.warning("No matching image-label pairs found.")
        return

    st.subheader("Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", len(os.listdir(image_dir)))
    col2.metric("Total Labels", len(os.listdir(label_dir)))
    col3.metric("Matched Pairs", len(matched_pairs))

    selected_pairs = random.sample(matched_pairs, min(num_samples, len(matched_pairs)))
    cmap = LinearSegmentedColormap.from_list("segmentation", [(0, 0, 0, 0), (1, 0, 0, 0.7)], N=2)

    for idx, (img_file, lbl_file) in enumerate(selected_pairs):
        st.markdown(f"### Sample {idx+1}: {img_file}")
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, lbl_file)

        img_data = nib.load(img_path).get_fdata()
        lbl_data = nib.load(lbl_path).get_fdata()

        if img_data.shape != lbl_data.shape:
            st.warning(f"Shape mismatch: Image {img_data.shape} vs Label {lbl_data.shape}")
            continue

        display_statistics(img_data, lbl_data)

        # Fixed slice and orientation
        orientation = "Axial"
        axis = 2
        slice_idx = img_data.shape[axis] // 2
        view_mode = "Overlay"

        st.markdown(f"**View:** {orientation}, Slice: {slice_idx}, Mode: {view_mode}")
        visualize_slice(img_data, lbl_data, axis, slice_idx, view_mode, cmap)
        st.markdown("---")


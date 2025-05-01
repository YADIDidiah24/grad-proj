import streamlit as st
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Import utility functions
from utils import (
    download_and_extract_kaggle_dataset,
    find_image_and_label_folders,
    explore_dataset
)

from models import (
    visualize_model_architecture
)

from training import (run_segmentation_pipeline)
from metrics import (visualize_slices, evaluate_model)

# Configure the Streamlit page
st.set_page_config(
    page_title="3D Medical Image Segmentation",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom right, #1f2937, #111827);
        color: white;
    }
    .header {
        background: linear-gradient(to right, #65a30d, #1d4ed8, #4b5563);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar {
        background: linear-gradient(to bottom, #1d4ed8, #1e3a8a);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .content {
        background: linear-gradient(to right, #111827, #1f2937);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        background: linear-gradient(to right, #1d4ed8, #6d28d9);
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .btn {
        background-color: #1d4ed8;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
    }
    .btn:hover {
        background-color: #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application title and description
st.markdown(
    """
    <div class="header">
        <h1>3D Medical Image Segmentation</h1>
        <p>This application allows you to train a 3D V-Net model for medical image segmentation.
        Upload a dataset URL containing NIFTI format images and labels to begin the training process.
        Once training is complete, you can download the trained model and view the performance metrics.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state initialization
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'sample_prediction' not in st.session_state:
    st.session_state.sample_prediction = None
if 'dataset_processed' not in st.session_state:
    st.session_state.dataset_processed = False
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame()  # Initialize as an empty DataFrame
# Add these new session state variables to store paths
if 'image_dir' not in st.session_state:
    st.session_state.image_dir = None
if 'label_dir' not in st.session_state:
    st.session_state.label_dir = None
if 'all_trained_models' not in st.session_state:
    st.session_state.all_trained_models = {}

# Sidebar for navigation
st.sidebar.markdown(
    """
    <div class="sidebar">
        <h2>Navigation</h2>
        <ul>
            <li><a href="#dataset-input">Dataset Input</a></li>
            <li><a href="#model-selection">Model Selection</a></li>
            <li><a href="#training-parameters">Training Parameters</a></li>
            <li><a href="#training-results">Training Results</a></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Dataset URL input
st.markdown("<h2 id='dataset-input'>Dataset Input</h2>", unsafe_allow_html=True)
dataset_url = st.text_input(
    "Enter URL of the dataset zip file containing NIFTI images and labels",
    help="The dataset should contain two directories: 'images' and 'labels' with corresponding NIFTI files"
)

# Handle dataset download and processing
if dataset_url and not st.session_state.dataset_processed:
    st.markdown("<h2>Dataset Processing</h2>", unsafe_allow_html=True)

    if "kaggle.com" in dataset_url:
        try:
            with st.expander("Processing Details", expanded=True):
                # Call the enhanced function that now includes progress bars
                dataset_path = download_and_extract_kaggle_dataset(dataset_url)

                # Find image and label directories
                with st.spinner("Locating image and label folders..."):
                    image_dir, label_dir = find_image_and_label_folders(dataset_path)

                    # Store paths in session state
                    st.session_state.image_dir = image_dir
                    st.session_state.label_dir = label_dir

                # Mark dataset as processed to avoid re-downloading
                st.session_state.dataset_processed = True

                # Display success message and paths
                st.success(f"✅ Dataset successfully processed! Path: {dataset_path}")

                # Show details about the directories
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Images folder:** `{image_dir}`")
                    if os.path.exists(image_dir):
                        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))]
                        st.write(f"Found {len(image_files)} image files")

                with col2:
                    st.info(f"**Labels folder:** `{label_dir}`")
                    if os.path.exists(label_dir):
                        label_files = [f for f in os.listdir(label_dir) if f.endswith(('.nii', '.nii.gz'))]
                        st.write(f"Found {len(label_files)} label files")

        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
    else:
        st.warning("Currently only Kaggle dataset URLs are supported.")
elif st.session_state.dataset_processed:
    st.success("Dataset has been processed successfully!")

    # Reset button to allow downloading another dataset
    if st.button("Process another dataset", key="reset_dataset"):
        st.session_state.dataset_processed = False
        st.session_state.image_dir = None
        st.session_state.label_dir = None
        st.experimental_rerun()

# Dataset exploration section before model selection
if st.session_state.dataset_processed and st.session_state.image_dir and st.session_state.label_dir:
    st.markdown("<h2>🔍 Explore Your Dataset</h2>", unsafe_allow_html=True)
    with st.expander("Click to explore your dataset (image/label previews and statistics)", expanded=False):
        try:
            explore_dataset(st.session_state.image_dir, st.session_state.label_dir)
        except Exception as e:
            st.error(f"An error occurred during dataset exploration: {e}")

# Model Selection
st.markdown("<h2 id='model-selection'>Model Selection</h2>", unsafe_allow_html=True)
model_options = {
    "3D U-Net": "The 3D adaptation of the popular U-Net architecture, with encoder-decoder structure and skip connections.",
    "3D V-Net": "A 3D convolutional network with skip connections and residual blocks, optimized for volumetric medical segmentation.",
    "SA-Net": "Utilizes self-attention mechanisms to capture long-range dependencies in 3D volumes.",
    "E1D3 U-Net": "An enhanced 1D-3D U-Net architecture, likely combining 1D and 3D convolutional pathways for improved feature extraction.",
    "HDC Net": "A network employing Hybrid Dilated Convolutions to capture multi-scale contextual information effectively in 3D data."
}

model_name_to_key = {
    "3D U-Net": "unet",
    "3D V-Net": "vnet",
    "SA-Net": "sanet",
    "E1D3 U-Net": "e1d3_unet",
    "HDC Net": "hdcnet"
}

col1, col2 = st.columns([2, 3])
with col1:
    selected_models = st.multiselect(
        "Select Models to Train",
        default=["3D V-Net"],
        options=list(model_options.keys()),
        help="Choose one or more models to train. Selecting multiple models allows for comparison."
    )

    if not selected_models:
        st.warning("Please select at least one model for training.")

with col2:
    if selected_models:
        for model in selected_models:
            st.info(f"**{model}**: {model_options[model]}")

# Model architecture visualization
st.markdown("<h2>Model Architecture Visualization</h2>", unsafe_allow_html=True)
visualize_button = st.button("Visualize Model Architecture")

if visualize_button:
    if st.session_state.model:
        visualize_model_architecture(st.session_state.model)
    else:
        st.warning("No model is available for visualization. Please train or load a model first.")

# Training parameters
st.markdown("<h2 id='training-parameters'>Training Parameters</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=1)
with col2:
    test_size = st.slider("Test Split Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

target_shape = (64, 64, 64)

# Start training button - only enable if dataset is processed
start_training = st.button("Start Training", disabled=(not st.session_state.dataset_processed))

# Create a placeholder for training information
training_info = st.empty()

# Training logic
BATCH_SIZE = 2
if start_training and st.session_state.image_dir and st.session_state.label_dir:
    with st.spinner("Training in progress..."):
        data_params = {
            'image_path': st.session_state.image_dir,
            'label_path': st.session_state.label_dir,
            'test_size': test_size
        }

        training_params = {
            'epochs': epochs,
            'batch_size': BATCH_SIZE
        }

        model_keys_to_train = [model_name_to_key[model] for model in selected_models]

        specific_model_params = {
            'unet': {
                'target_shape': target_shape,
                'use_generator': False,
                'use_dice_loss': True
            },
            'vnet': {
                'target_shape': target_shape,
                'use_generator': True,
                'use_dice_loss': True
            },
            'sanet': {
                'target_shape': target_shape,
                'use_generator': False,
                'use_dice_loss': True,
                'num_filters': 16
            },
            'e1d3_unet': {
                'target_shape': target_shape,
                'use_generator': False,
                'use_dice_loss': True,
                'num_filters': 16
            },
            'hdcnet': {
                'target_shape': target_shape,
                'use_generator': True,
                'use_dice_loss': True
            }
        }

        progress_bar = st.progress(0)

        def update_progress(progress):
            progress_bar.progress(progress)

        comparison_df, all_trained_models = run_segmentation_pipeline(
            models_to_train=model_keys_to_train,
            data_paths=data_params,
            training_params=training_params,
            specific_model_params=specific_model_params,
            save_models=True,
            compare=True,
            visualize=True,
            progress_callback=update_progress
        )

        st.session_state.training_complete = True
        st.session_state.comparison_df = comparison_df
        st.session_state.all_trained_models = all_trained_models

        if all_trained_models:
            first_model_key = model_keys_to_train[0]
            if first_model_key in all_trained_models:
                st.session_state.model = all_trained_models[first_model_key]['model']
                st.session_state.metrics = all_trained_models[first_model_key]['metrics']
                st.session_state.history = all_trained_models[first_model_key]['history']
                st.session_state.sample_prediction = all_trained_models[first_model_key]['test_data']
                st.success("Training completed successfully!")
            else:
                st.error(f"Model {first_model_key} not found in trained models.")
        else:
            st.error("No models were trained successfully.")
elif start_training and (not st.session_state.image_dir or not st.session_state.label_dir):
    st.error("Image or label directories not found. Please ensure the dataset is properly processed.")

# Display training results if training is complete
if st.session_state.training_complete:
    st.markdown("<h2 id='training-results'>Training Results</h2>", unsafe_allow_html=True)

    comparison_df_reset = st.session_state.comparison_df.reset_index()
    st.write("### Model Comparison")
    st.dataframe(comparison_df_reset)

    metrics_to_plot = ["Dice Score", "Accuracy", "F1 Score", "Precision", "Recall"]
    st.write("### Model Performance Metrics Comparison")

    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.15
    index = np.arange(len(comparison_df_reset['Model']))

    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df_reset[metric].values
        bars = ax.bar(index + i * bar_width, values, bar_width, label=metric)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, round(yval, 2), ha='center', va='bottom')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics Comparison')
    ax.set_xticks(index + bar_width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(comparison_df_reset['Model'])
    ax.legend(title='Metric')
    ax.set_ylim(0, 1.05)

    st.pyplot(fig)

    selected_model_for_results = st.selectbox("Select Model to View Results", options=selected_models)
    selected_model_key = model_name_to_key[selected_model_for_results]

    st.write("### Model Metrics")
    if selected_model_key in st.session_state.all_trained_models:
        st.write(st.session_state.all_trained_models[selected_model_key]['metrics'])
    else:
        st.warning(f"Metrics for {selected_model_for_results} not available.")

    st.write("### Model Loss and Accuracy Comparison")
    if selected_model_key in st.session_state.all_trained_models:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(st.session_state.all_trained_models[selected_model_key]['history'].history['loss'], label='Training Loss')
        ax1.plot(st.session_state.all_trained_models[selected_model_key]['history'].history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(st.session_state.all_trained_models[selected_model_key]['history'].history['accuracy'], label='Training Accuracy')
        ax2.plot(st.session_state.all_trained_models[selected_model_key]['history'].history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        st.pyplot(fig)
    else:
        st.warning(f"Loss and Accuracy Comparison for {selected_model_for_results} not available.")

    st.write("### Sample Prediction")
    if selected_model_key in st.session_state.all_trained_models:
        X_test, y_test, y_pred, y_pred_prob = st.session_state.all_trained_models[selected_model_key]['test_data']

        st.write(f"X_test shape: {X_test.shape}")
        st.write(f"y_test shape: {y_test.shape}")
        st.write(f"y_pred shape: {y_pred.shape}")
        st.write(f"y_pred_prob shape: {y_pred_prob.shape}")

        try:
            sample_idx = 0
            slice_idx = X_test.shape[1] // 2

            if len(X_test.shape) == 5:
                x_sample = X_test[sample_idx, :, :, :, 0]
                y_true_sample = y_test[sample_idx, :, :, :, 0]
                y_pred_sample = y_pred[sample_idx, :, :, :, 0]

                fig = visualize_slices(
                    x_sample,
                    y_true_sample,
                    y_pred_sample,
                    slice_index=slice_idx,
                    model_name=selected_model_for_results.upper()
                )
                st.pyplot(fig)

                st.write("### Additional Slice Views")
                col1, col2 = st.columns(2)

                with col1:
                    sagittal_slice = X_test.shape[2] // 2
                    fig_sagittal = visualize_slices(
                        x_sample,
                        y_true_sample,
                        y_pred_sample,
                        slice_index=sagittal_slice,
                        model_name=f"{selected_model_for_results.upper()} - Sagittal View"
                    )
                    st.pyplot(fig_sagittal)

                with col2:
                    coronal_slice = X_test.shape[3] // 2
                    fig_coronal = visualize_slices(
                        np.transpose(x_sample, (1, 0, 2)),
                        np.transpose(y_true_sample, (1, 0, 2)),
                        np.transpose(y_pred_sample, (1, 0, 2)),
                        slice_index=coronal_slice,
                        model_name=f"{selected_model_for_results.upper()} - Coronal View"
                    )
                    st.pyplot(fig_coronal)

            else:
                st.error("The data doesn't have the expected 5D shape for batch of 3D volumes with channels.")
        except Exception as e:
            st.error(f"Error visualizing slices: {str(e)}")
            st.write("Attempting alternative visualization approach...")

            try:
                if len(X_test.shape) >= 4:
                    sample_idx = 0
                    slice_idx = X_test.shape[1] // 2

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                    if len(X_test.shape) == 5:
                        axes[0].imshow(X_test[sample_idx, slice_idx, :, :, 0], cmap="gray")
                        axes[1].imshow(y_test[sample_idx, slice_idx, :, :, 0], cmap="gray")
                        axes[2].imshow(y_pred[sample_idx, slice_idx, :, :, 0], cmap="gray")
                    elif len(X_test.shape) == 4:
                        axes[0].imshow(X_test[sample_idx, slice_idx], cmap="gray")
                        axes[1].imshow(y_test[sample_idx, slice_idx], cmap="gray")
                        axes[2].imshow(y_pred[sample_idx, slice_idx], cmap="gray")

                    axes[0].set_title("Input Image")
                    axes[1].set_title("True Label")
                    axes[2].set_title("Predicted Label")

                    for ax in axes:
                        ax.axis("off")

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error("Data shape is not compatible with visualization.")
            except Exception as e:
                st.error(f"Alternative visualization also failed: {str(e)}")
    else:
        st.warning(f"Sample prediction for {selected_model_for_results} not available.")

# Footer
import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
        .footer {
            background: linear-gradient(to right, #1e3a8a, #6b21a8);
            color: #d1d5db;
            padding: 2rem;
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
        }
        .footer h3 {
            margin-bottom: 0.5rem;
            font-size: 1.25rem;
            font-weight: 600;
        }
        .footer p {
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }
        .footer-links, .footer-icons {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }
        .footer-link {
            color: #d1d5db;
            text-decoration: none;
            font-size: 0.875rem;
        }
        .footer-link:hover {
            color: #2dd4bf;
        }
        .footer-icon {
            height: 32px;
            width: 32px;
        }
    </style>
""", unsafe_allow_html=True)

# Footer HTML
st.markdown("""
    <div class="footer">
        <h3>AI Model Trainer</h3>
        <p>Revolutionizing AI model training, one step at a time.</p>
        <div class="footer-links">
            <a href="#faq" class="footer-link">FAQs</a>
            <a href="#docs" class="footer-link">Docs</a>
            <a href="#support" class="footer-link">Support</a>
            <a href="#terms" class="footer-link">Terms</a>
        </div>
        <div class="footer-icons">
            <a href="https://stackoverflow.com" class="footer-link">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" alt="Stack Overflow" class="footer-icon">
            </a>
            <a href="https://linkedin.com" class="footer-link">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/e8/Linkedin-logo-blue-In-square-40px.png" alt="LinkedIn" class="footer-icon">
            </a>
            <a href="https://github.com" class="footer-link">
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" class="footer-icon">
            </a>
            <a href="https://linkedin.com" class="footer-link">
                <img src="https://img.icons8.com/?size=100&id=2969&format=png&color=000000" alt="LinkedIn Alt" class="footer-icon">
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)


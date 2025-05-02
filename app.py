import streamlit as st
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import io
from fpdf import FPDF
import pickle
import joblib
import io
import tempfile
from tensorflow.keras.models import save_model
import csv 
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
from metrics import (visualize_slices, visualize_3d_volume, visualize_3d_comparison)

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

# Set Seaborn style to match the app's color scheme
sns.set(style="darkgrid", palette=["#65a30d", "#1d4ed8", "#4b5563", "#1f2937", "#111827"])

# Function to add rounded borders to Matplotlib plots
def add_rounded_borders(ax, border_color='white', border_width=2, radius=10):
    # Create a rounded rectangle patch
    from matplotlib.patches import FancyBboxPatch
    bb = ax.get_window_extent()
    width, height = bb.width, bb.height
    rect = FancyBboxPatch((bb.x0, bb.y0), width, height,
                          boxstyle=f"round,pad={border_width},rounding_size={radius}",
                          edgecolor=border_color, facecolor='none', linewidth=border_width)
    ax.add_patch(rect)
    ax.set_clip_on(False)

# Function to create a PDF report


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
    <style>
    .sidebar {
        background: linear-gradient(to bottom, #1e3a8a, #134e4a, #0f172a);
        padding: 24px 20px;
        border-radius: 10px;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    .sidebar h2 {
        font-size: 22px;
        margin-bottom: 20px;
        font-weight: 600;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        padding-bottom: 8px;
    }

    .sidebar ul {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }

    .sidebar li {
        margin: 12px 0;
    }

    .sidebar a {
        color: white;
        text-decoration: none;
        font-size: 16px;
        transition: color 0.2s ease, padding-left 0.2s ease;
        display: block;
    }

    .sidebar a:hover {
        color: #5eead4;  /* Tailwind teal-300 */
        padding-left: 4px;
    }
    </style>

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

            # Add the visualization button next to the model description
            if st.button(f"Visualize {model} Architecture", key=f"viz_{model}"):
                st.subheader(f"Architecture of {model}")
                visualize_model_architecture(model)

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
                'use_generator': True,
                'use_dice_loss': True
            },
            'vnet': {
                'target_shape': target_shape,
                'use_generator': True,
                'use_dice_loss': True
            },
            'sanet': {
                'target_shape': target_shape,
                'use_generator': True,
                'use_dice_loss': True,
                'num_filters': 16
            },
            'e1d3_unet': {
                'target_shape': target_shape,
                'use_generator': True,
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
# Display training results if training is complete
if st.session_state.training_complete:
    st.markdown("<h2 id='training-results'>Training Results</h2>", unsafe_allow_html=True)

    comparison_df_reset = st.session_state.comparison_df.reset_index()
    st.write("### Model Comparison")
    st.dataframe(comparison_df_reset)

    csv_data = comparison_df_reset.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Model Comparison as CSV",
        data=csv_data,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

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

    # Add rounded borders to the plot
    add_rounded_borders(ax)

    st.pyplot(fig)

    selected_model_for_results = st.selectbox("Select Model to View Results", options=selected_models)
    selected_model_key = model_name_to_key[selected_model_for_results]

    st.write("### Model Metrics")
    if selected_model_key in st.session_state.all_trained_models:
        st.write(st.session_state.all_trained_models[selected_model_key]['metrics'])
    else:
        st.warning(f"Metrics for {selected_model_for_results} not available.")

    st.write("### Model Loss and Accuracy Comparison")
    max_epochs = 20
    # Ensure session state and model key exist
    if 'all_trained_models' in st.session_state and selected_model_key in st.session_state.all_trained_models:
        model_data = st.session_state.all_trained_models[selected_model_key]
        history = model_data.get('history', None)

        if history is not None and hasattr(history, 'history'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            ax1.plot(history.history.get('loss', [])[:max_epochs], label='Training Loss')
            ax1.plot(history.history.get('val_loss', [])[:max_epochs], label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()

            ax1.set_ylim(0, 5)

            add_rounded_borders(ax1)

            ax2.plot(history.history.get('accuracy', [])[:max_epochs], label='Training Accuracy')
            ax2.plot(history.history.get('val_accuracy', [])[:max_epochs], label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()

            ax2.set_ylim(0.9, 1.0)
            add_rounded_borders(ax2)

            st.pyplot(fig)
        else:
            st.warning(f"Training history for `{selected_model_key}` not available or invalid.")
    else:
        st.warning(f"Loss and Accuracy Comparison for `{selected_model_key}` not available.")

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

            # Handle both 5D (B, D, H, W, C) and 4D (B, D, H, W)
            if len(X_test.shape) == 5:
                x_sample = X_test[sample_idx, :, :, :, 0]
                y_true_sample = y_test[sample_idx, :, :, :, 0]
                y_pred_sample = y_pred[sample_idx, :, :, :, 0]
            elif len(X_test.shape) == 4:
                x_sample = X_test[sample_idx]
                y_true_sample = y_test[sample_idx]
                y_pred_sample = y_pred[sample_idx]
            else:
                raise ValueError(f"Unsupported data shape: {X_test.shape}")

            # Axial view
            fig_axial = visualize_slices(
                x_sample,
                y_true_sample,
                y_pred_sample,
                slice_index=slice_idx,
                model_name=selected_model_for_results.upper()
            )
            # Add rounded borders to the plot
            add_rounded_borders(fig_axial.axes[0])
            st.pyplot(fig_axial)

            # Additional slice views
            st.write("### Additional Slice Views")
            col1, col2 = st.columns(2)

            # Sagittal view
            with col1:
                sagittal_idx = x_sample.shape[1] // 2
                fig_sagittal = visualize_slices(
                    np.transpose(x_sample, (1, 0, 2)),
                    np.transpose(y_true_sample, (1, 0, 2)),
                    np.transpose(y_pred_sample, (1, 0, 2)),
                    slice_index=sagittal_idx,
                    model_name=f"{selected_model_for_results.upper()} - Sagittal View"
                )
                # Add rounded borders to the plot
                add_rounded_borders(fig_sagittal.axes[0])
                st.pyplot(fig_sagittal)

            # Coronal view
            with col2:
                coronal_idx = x_sample.shape[2] // 2
                fig_coronal = visualize_slices(
                    np.transpose(x_sample, (2, 0, 1)),
                    np.transpose(y_true_sample, (2, 0, 1)),
                    np.transpose(y_pred_sample, (2, 0, 1)),
                    slice_index=coronal_idx,
                    model_name=f"{selected_model_for_results.upper()} - Coronal View"
                )
                # Add rounded borders to the plot
                add_rounded_borders(fig_coronal.axes[0])
                st.pyplot(fig_coronal)

        except Exception as e:
            st.error(f"Error visualizing slices: {str(e)}")
    else:
        st.warning(f"Sample prediction for `{selected_model_for_results}` not available.")

    # Add necessary imports at the top of app.py

    # Add 3D visualization
    st.write("### 3D Volume Visualization")

    with st.expander(f" ## {selected_model_for_results} - 3D Volume Visualization", expanded=False):
        try:
            # Create 3D visualization
            fig_3d = visualize_3d_comparison(
                x_sample,
                y_true_sample,
                y_pred_sample,
                model_name=selected_model_for_results.upper()
            )

            # Display the interactive 3D visualization
            st.plotly_chart(fig_3d, use_container_width=True)

            # Add a note about interaction
            st.info("⚠️ **Tip**: Click and drag to rotate the 3D view. Use the buttons above to toggle visibility of different components.")

            # Option to view individual components
            st.write("#### Individual 3D Components")
            tab1, tab2, tab3 = st.tabs(["Volume", "True Label", "Predicted Label"])

            with tab1:
                fig_vol, _, _ = visualize_3d_volume(
                    x_sample,
                    y_true_sample,
                    y_pred_sample,
                    model_name=selected_model_for_results.upper()
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            with tab2:
                _, fig_true, _ = visualize_3d_volume(
                    x_sample,
                    y_true_sample,
                    y_pred_sample,
                    model_name=selected_model_for_results.upper()
                )
                st.plotly_chart(fig_true, use_container_width=True)

            with tab3:
                _, _, fig_pred = visualize_3d_volume(
                    x_sample,
                    y_true_sample,
                    y_pred_sample,
                    model_name=selected_model_for_results.upper()
                )
                st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating 3D visualization: {str(e)}")
            st.info("3D visualization requires sufficient segmentation data to create surfaces. If this error persists, try a different sample or model.")

    # Add a download option for the 3D figures (optional)
    with st.expander("Export 3D Visualizations", expanded=False):
        try:
            col1, col2 = st.columns(2)

            with col1:
                # Generate HTML file with the 3D visualization
                fig_3d_export = visualize_3d_comparison(
                    x_sample,
                    y_true_sample,
                    y_pred_sample,
                    model_name=selected_model_for_results.upper()
                )
                html_bytes = fig_3d_export.to_html(include_plotlyjs='cdn').encode()
                st.download_button(
                    label="Download 3D Visualization (HTML)",
                    data=html_bytes,
                    file_name=f"{selected_model_for_results}_3d_viz.html",
                    mime="text/html"
                )

            with col2:
                # Option to adjust threshold for 3D visualization
                new_threshold = st.slider(
                    "Segmentation Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.05,
                    help="Adjust threshold for binary segmentation in 3D visualization"
                )

                if st.button("Regenerate with New Threshold"):
                    fig_3d_threshold = visualize_3d_comparison(
                        x_sample,
                        y_true_sample,
                        y_pred_sample,
                        threshold=new_threshold,
                        model_name=f"{selected_model_for_results.upper()} (Threshold: {new_threshold})"
                    )
                    st.plotly_chart(fig_3d_threshold, use_container_width=True)

        except Exception as e:
            st.error(f"Error with 3D export: {str(e)}")

    # Add a download option for the trained models
    st.write("### Download Trained Models")

SUPPORTED_FORMATS = {
    "Pickle (.pkl)": "pkl",
    "Joblib (.joblib)": "joblib",
    "HDF5 (.h5)": "h5",
    "Keras (.keras)": "keras"
}

for model_name, model_data in st.session_state.all_trained_models.items():
    model = model_data['model']

    st.write(f"#### {model_name}")

    selected_format = st.selectbox(
        f"Choose format for {model_name}",
        options=list(SUPPORTED_FORMATS.keys()),
        key=f"{model_name}_format_select"
    )

    format_ext = SUPPORTED_FORMATS[selected_format]

    # Serialize model based on selected format
    if format_ext in ["pkl", "joblib"]:
        buffer = io.BytesIO()
        if format_ext == "pkl":
            pickle.dump(model, buffer)
        else:
            joblib.dump(model, buffer)
        model_bytes = buffer.getvalue()

    elif format_ext in ["h5", "keras"]:
        # Check if it's a Keras model
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_ext}") as tmp_file:
                save_model(model, tmp_file.name, save_format=format_ext)
                tmp_file.seek(0)
                model_bytes = tmp_file.read()
            os.unlink(tmp_file.name)
        except Exception as e:
            st.warning(f"Could not export {model_name} as {format_ext.upper()}: {e}")
            continue
    else:
        st.warning(f"Unsupported format: {format_ext}")
        continue

    model_filename = f"{model_name}_model.{format_ext}"

    st.download_button(
        label=f"Download {model_name} as {format_ext.upper()}",
        data=model_bytes,
        file_name=model_filename,
        mime="application/octet-stream"
    )



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
        <h3 class="footer-title">AI Model Trainer <span class="icon">🧠🤖</span></h3>
        <p>Empowering researchers and practitioners with cutting-edge tools for 3D medical image segmentation. Our platform streamlines the process of training and evaluating AI models, making it accessible and efficient. Join us in revolutionizing medical imaging, one model at a time.</p>
        <div class="footer-links">
            <a href="#faq" class="footer-link">FAQs</a>
            <a href="#docs" class="footer-link">Documentation</a>
            <a href="#support" class="footer-link">Support</a>
            <a href="#terms" class="footer-link">Terms of Service</a>
            <a href="#privacy" class="footer-link">Privacy Policy</a>
        </div>
        <div class="footer-icons">
            <a href="https://stackoverflow.com" class="footer-link" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" alt="Stack Overflow" class="footer-icon">
            </a>
            <a href="https://www.linkedin.com/in/yadidiah-kanaparthi/" class="footer-link" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/e/e8/Linkedin-logo-blue-In-square-40px.png" alt="LinkedIn" class="footer-icon">
            </a>
            <a href="https://github.com/YADIDidiah24/grad-proj/tree/main" class="footer-link" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" class="footer-icon">
            </a>
            <a href="https://gamma.app/docs/Liver-Tumor-Segmentation-Comparing-Advanced-Neural-Network-Archit-qtx78j0tchkz1dz" class="footer-link">
                <img src="https://img.icons8.com/?size=100&id=2969&format=png&color=000000" alt="Presentation" class="footer-icon">
            </a>
        </div>
        <p class="footer-copyright">© 2025 AI Model Trainer. All rights reserved.</p>
    </div>
    <style>
        .footer-title {
            display: flex;
            align-items: center;
            font-size: 1.75rem;
            font-weight: 700;
            color: #d1d5db;
            margin-bottom: 0.5rem;
        }
        .icon {
            margin-left: 0.5rem;
            font-size: 1.5rem;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

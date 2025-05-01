import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, Add, concatenate, Input,
    Cropping3D, BatchNormalization, Dropout, ZeroPadding3D, GlobalAveragePooling3D,
    GlobalMaxPooling3D, Reshape, Multiply, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import streamlit as st
import os
from metrics import dice_coefficient, combined_loss



def ensure_shape_match(tensor, target_shape):
    current_shape = K.int_shape(tensor)

    for i in range(1, 4):  # Dimensions 1, 2, 3 (H, W, D)
        if current_shape[i] > target_shape[i]:
            # Need to crop
            diff = current_shape[i] - target_shape[i]
            crop_start = diff // 2
            crop_end = diff - crop_start
            crop_config = [(0, 0)] * 5  # Initialize with no cropping
            crop_config[i] = (crop_start, crop_end)  # Set cropping for dimension i

            tensor = Cropping3D(cropping=crop_config[1:4])(tensor)

        elif current_shape[i] < target_shape[i]:
            diff = target_shape[i] - current_shape[i]
            pad_start = diff // 2
            pad_end = diff - pad_start
            pad_config = [(0, 0)] * 5  # Initialize with no padding
            pad_config[i] = (pad_start, pad_end)  # Set padding for dimension i

            tensor = ZeroPadding3D(padding=pad_config[1:4])(tensor)
    return tensor


def build_3d_unet(input_shape=(64, 64, 64, 1), use_dice_loss=False):
    inputs = Input(input_shape)
    c1 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(p1)
    c2 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    # Bottleneck
    c3 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(p2)
    c3 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(c3)

    # Decoder
    u4 = UpSampling3D((2, 2, 2))(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(u4)
    c4 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(c4)

    u5 = UpSampling3D((2, 2, 2))(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(u5)
    c5 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(c5)

    outputs = Conv3D(1, (1, 1, 1), activation="sigmoid")(c5)

    model = Model(inputs, outputs)

    loss_fn = combined_loss if use_dice_loss else 'binary_crossentropy'
    metrics = [dice_coefficient, 'accuracy'] if use_dice_loss else ['accuracy']

    model.compile(optimizer=Adam(), loss=loss_fn, metrics=metrics)
    return model

def vnet_block(x, filters):
    res = Conv3D(filters, (3, 3, 3), padding="same", activation="relu")(x)
    res = Conv3D(filters, (3, 3, 3), padding="same", activation="relu")(res)
    return Add()([x, res])

def build_3d_vnet(input_shape=(64, 64, 64, 1), use_dice_loss=False):
    
    inputs = Input(input_shape)

    x = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(inputs)

    # Encoder
    e1 = vnet_block(x, 16)
    e1_shape = K.int_shape(e1)
    e2 = Conv3D(32, (3, 3, 3), strides=2, padding="same", activation="relu")(e1)
    e2 = vnet_block(e2, 32)
    e2_shape = K.int_shape(e2)
    e3 = Conv3D(64, (3, 3, 3), strides=2, padding="same", activation="relu")(e2)
    e3 = vnet_block(e3, 64)
    
    b = vnet_block(e3, 64)

    d3 = Conv3DTranspose(32, (3, 3, 3), strides=2, padding="same", activation="relu")(b)

    # Check if shapes need to be adjusted
    d3_shape = K.int_shape(d3)
    if d3_shape[1:4] != e2_shape[1:4]:
        # Crop or pad d3 to match e2's spatial dimensions
        d3 = ensure_shape_match(d3, e2_shape)

    d3 = Add()([d3, e2])  # Now both have 32 filters
    d3 = vnet_block(d3, 32)

    # Second upsampling - from 32 to 16 filters to match e1
    d2 = Conv3DTranspose(16, (3, 3, 3), strides=2, padding="same", activation="relu")(d3)

    # Check if shapes need to be adjusted
    d2_shape = K.int_shape(d2)
    if d2_shape[1:4] != e1_shape[1:4]:
        # Crop or pad d2 to match e1's spatial dimensions
        d2 = ensure_shape_match(d2, e1_shape)

    d2 = Add()([d2, e1])  # Now both have 16 filters
    d2 = vnet_block(d2, 16)

    outputs = Conv3D(1, (1, 1, 1), activation="sigmoid")(d2)

    model = Model(inputs, outputs)

    # Choose appropriate loss function
    loss_fn = combined_loss if use_dice_loss else 'binary_crossentropy'
    metrics = [dice_coefficient, 'accuracy'] if use_dice_loss else ['accuracy']

    model.compile(optimizer=Adam(), loss=loss_fn, metrics=metrics)
    return model


def spatial_attention_block(x):
    # Get channel count from input
    channels = K.int_shape(x)[-1]

    # Global Pooling
    avg_pool = GlobalAveragePooling3D()(x)
    max_pool = GlobalMaxPooling3D()(x)

    # Reshape for concatenation - need to match dimensions for multiplication later
    avg_pool = Reshape((1, 1, 1, channels))(avg_pool)
    max_pool = Reshape((1, 1, 1, channels))(max_pool)

    # Concatenate along channel axis
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # Apply 3D convolution to generate attention weights
    attention = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(concat)
    
    # Multiply attention weights with input tensor
    return Multiply()([x, attention])


def build_sa_net(input_shape=(64, 64, 64, 1), use_dice_loss=False):
    # Check if input shape is 4D or 5D and adjust accordingly
    if len(input_shape) == 4:  # (h, w, d, c)
        inputs = Input(input_shape)
    elif len(input_shape) == 5:  # (batch, h, w, d, c)
        inputs = Input(input_shape[1:])  # Remove batch dimension
    else:
        raise ValueError(f"Expected input shape with 4 or 5 dimensions, got {len(input_shape)}")

    # Encoder
    e1 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(inputs)
    e1 = spatial_attention_block(e1)
    e1 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(e1)

    e2 = Conv3D(32, (3, 3, 3), strides=2, padding="same", activation="relu")(e1)
    e2 = spatial_attention_block(e2)
    e2 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(e2)

    e3 = Conv3D(64, (3, 3, 3), strides=2, padding="same", activation="relu")(e2)
    e3 = spatial_attention_block(e3)
    e3 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(e3)

    # Bottleneck
    b = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(e3)
    b = spatial_attention_block(b)
    b = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(b)

    # Decoder
    d3 = Conv3DTranspose(64, (3, 3, 3), strides=2, padding="same", activation="relu")(b)
    d3 = Concatenate()([d3, e2])  # Use Concatenate instead of Add
    d3 = spatial_attention_block(d3)
    d3 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(d3)

    d2 = Conv3DTranspose(32, (3, 3, 3), strides=2, padding="same", activation="relu")(d3)
    d2 = Concatenate()([d2, e1])  # Use Concatenate instead of Add
    d2 = spatial_attention_block(d2)
    d2 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(d2)

    d1 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(d2)  # Keep this as is
    d1 = spatial_attention_block(d1)
    d1 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(d1)

    outputs = Conv3D(1, (1, 1, 1), activation="sigmoid")(d1)

    # Define and compile the model
    model = Model(inputs, outputs)

    loss_fn = combined_loss if use_dice_loss else 'binary_crossentropy'
    metrics = [dice_coefficient, 'accuracy'] if use_dice_loss else ['accuracy']

    model.compile(optimizer=Adam(), loss=loss_fn, metrics=metrics)
    return model

def e1d3_block(x, filters):
    
    res = Conv3D(filters, (3, 3, 3), padding="same", activation="relu")(x)
    res = Conv3D(filters, (3, 3, 3), padding="same", activation="relu")(res)
    return Add()([x, res])

def build_e1d3_unet(input_shape=(64, 64, 64, 1), use_dice_loss=False, num_filters=16):
    inputs = Input(input_shape)

    # Encoder Path
    conv1 = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(inputs)
    e1d3_1 = e1d3_block(conv1, num_filters)
    pool1 = Conv3D(num_filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(e1d3_1)

    conv2 = Conv3D(num_filters * 2, (3, 3, 3), activation='relu', padding='same')(pool1)
    e1d3_2 = e1d3_block(conv2, num_filters * 2)
    pool2 = Conv3D(num_filters * 2, (3, 3, 3), strides=(2, 2, 2), padding='same')(e1d3_2)

    conv3 = Conv3D(num_filters * 4, (3, 3, 3), activation='relu', padding='same')(pool2)
    e1d3_3 = e1d3_block(conv3, num_filters * 4)
    pool3 = Conv3D(num_filters * 4, (3, 3, 3), strides=(2, 2, 2), padding='same')(e1d3_3)

    # Bottleneck
    conv4 = Conv3D(num_filters * 8, (3, 3, 3), activation='relu', padding='same')(pool3)
    e1d3_4 = e1d3_block(conv4, num_filters * 8)

    # Decoder Path
    up5 = Conv3DTranspose(num_filters * 4, (3, 3, 3), strides=(2, 2, 2), padding='same')(e1d3_4)
    up5 = concatenate([up5, e1d3_3])
    conv5 = Conv3D(num_filters * 4, (3, 3, 3), activation='relu', padding='same')(up5)
    e1d3_5 = e1d3_block(conv5, num_filters * 4)

    up6 = Conv3DTranspose(num_filters * 2, (3, 3, 3), strides=(2, 2, 2), padding='same')(e1d3_5)
    up6 = concatenate([up6, e1d3_2])
    conv6 = Conv3D(num_filters * 2, (3, 3, 3), activation='relu', padding='same')(up6)
    e1d3_6 = e1d3_block(conv6, num_filters * 2)

    up7 = Conv3DTranspose(num_filters, (3, 3, 3), strides=(2, 2, 2), padding='same')(e1d3_6)
    up7 = concatenate([up7, e1d3_1])
    conv7 = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(up7)
    e1d3_7 = e1d3_block(conv7, num_filters)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(e1d3_7)

    model = Model(inputs=inputs, outputs=outputs)

    # Choose appropriate loss function
    loss_fn = combined_loss if use_dice_loss else 'binary_crossentropy'
    metrics = [dice_coefficient, 'accuracy'] if use_dice_loss else ['accuracy']

    model.compile(optimizer=Adam(), loss=loss_fn, metrics=metrics)
    return model


def hybrid_dilated_conv_block(x, filters, dilation_rates=[1, 2, 4]):
    residual = x
    x = BatchNormalization()(x)
    dilated_outputs = []
    for rate in dilation_rates:
        dilated_conv = Conv3D(
            filters // len(dilation_rates),
            kernel_size=(3, 3, 3),
            dilation_rate=(rate, rate, rate),
            padding='same',
            activation='relu'
        )(x)
        dilated_outputs.append(dilated_conv)

    x = Concatenate()(dilated_outputs)

    # Project back to original number of filters
    x = Conv3D(filters, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)

    if residual.shape[-1] != filters:
        residual = Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(residual)

    return Add()([x, residual])


def build_3d_hdc_net(input_shape=(64, 64, 64, 1), use_dice_loss=False):
    inputs = Input(input_shape)

    # Initial convolution
    x = Conv3D(16, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)

    # Encoder path
    enc1 = hybrid_dilated_conv_block(x, 16, dilation_rates=[1, 2, 3])
    pool1 = Conv3D(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(enc1)

    enc2 = hybrid_dilated_conv_block(pool1, 32, dilation_rates=[1, 2, 5])
    pool2 = Conv3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu')(enc2)

    # Bottleneck
    bottleneck = hybrid_dilated_conv_block(pool2, 64, dilation_rates=[1, 2, 4, 8])
    bottleneck = Dropout(0.3)(bottleneck)

    # Decoder path
    up1 = Conv3DTranspose(32, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(bottleneck)
    concat1 = concatenate([up1, enc2])  # Skip connection with enc2
    dec1 = hybrid_dilated_conv_block(concat1, 32, dilation_rates=[1, 2, 4])

    up2 = Conv3DTranspose(16, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(dec1)
    concat2 = concatenate([up2, enc1])  # Skip connection with enc1
    dec2 = hybrid_dilated_conv_block(concat2, 16, dilation_rates=[1, 2, 3])

    # Output layer
    outputs = Conv3D(1, kernel_size=(1, 1, 1), activation='sigmoid')(dec2)

    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs)

    # Choose appropriate loss function
    loss_fn = combined_loss if use_dice_loss else 'binary_crossentropy'
    metrics = [dice_coefficient, 'accuracy'] if use_dice_loss else ['accuracy']

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn, metrics=metrics)

    return model

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import streamlit as st

def visualize_model_architecture(model):
    """
    Visualizes the architecture of a given Keras model.

    Parameters:
    model (tf.keras.Model): The Keras model whose architecture needs to be visualized.
    """
    if model is None:
        st.error("No model available to visualize.")
        return

    # Generate the model architecture plot
    plot = plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

    # Display the plot in Streamlit
    st.image('model_architecture.png', caption='Model Architecture', use_column_width=True)

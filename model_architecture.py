"""
Phase 4: Model Development for Discrete-Time Survival Prediction

This module defines the 1D-CNN model architecture (ResNet-style)
and the custom survival loss function.

This file is separate from model_utils.py, which contains
data splitting and evaluation helpers.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Constants for the model ---
# We define them here so the model is self-contained
INPUT_LENGTH = 4096      # Number of time samples per ECG
N_LEADS = 8              # Number of ECG leads after Phase 2 selection
N_INTERVALS = 120        # 120 intervals (10 years, 1 per month)
BASE_FILTERS = 64        # Filters in the first block of the CNN
KERNEL_SIZE = 7          # Kernel size for ResNet blocks
EPSILON = 1e-07          # Small constant for loss function to prevent log(0)


# --- Model Helper Functions (ResNet Blocks) ---

def conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
    name: str | None = None,
) -> tf.Tensor:
    """Convolution + BatchNorm + ReLU helper block."""
    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else name + "_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    x = layers.Activation("relu", name=None if name is None else name + "_relu")(x)
    return x


def residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = KERNEL_SIZE,
    strides: int = 1,
    name: str | None = None,
) -> tf.Tensor:
    """A simple ResNet-style residual block for 1D signals."""
    shortcut = x

    # Main path
    x = conv_bn_relu(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        name=None if name is None else name + "_conv1",
    )
    x = layers.Conv1D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else name + "_conv2",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn2")(x)

    # Shortcut path (projection if needed)
    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv1D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=None if name is None else name + "_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(
            name=None if name is None else name + "_proj_bn"
        )(shortcut)

    # Add & ReLU
    x = layers.Add(name=None if name is None else name + "_add")([x, shortcut])
    x = layers.Activation("relu", name=None if name is None else name + "_out")(x)
    return x


# --- Model Build Function ---

def build_ecg_survival_model(
    n_intervals: int = N_INTERVALS,
    input_length: int = INPUT_LENGTH,
    n_leads: int = N_LEADS,
    base_filters: int = BASE_FILTERS,
) -> keras.Model:
    """Build the 1D-CNN model for ECG-based survival prediction."""
    print("\nBuilding 1D-CNN ResNet model...")
    inputs = keras.Input(shape=(input_length, n_leads), name="ecg_input")

    # Initial conv + max-pooling (like a small "stem")
    x = conv_bn_relu(inputs, filters=base_filters, kernel_size=7, strides=2, name="stem")
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name="stem_pool")(x)

    # Residual stages
    x = residual_block(x, filters=base_filters, strides=1, name="stage1_block1")
    x = residual_block(x, filters=base_filters, strides=1, name="stage1_block2")

    x = residual_block(x, filters=base_filters * 2, strides=2, name="stage2_block1")
    x = residual_block(x, filters=base_filters * 2, strides=1, name="stage2_block2")

    x = residual_block(x, filters=base_filters * 4, strides=2, name="stage3_block1")
    x = residual_block(x, filters=base_filters * 4, strides=1, name="stage3_block2")

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    outputs = layers.Dense(
        n_intervals,
        activation="sigmoid",
        name="survival_logits",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ecg_survival_cnn")
    print("Model build complete.")
    return model


# --- Custom Loss Function (MANUAL MATH FIX) ---

def survival_loss_with_mask(y_true_masked: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Manually calculates the masked binary cross-entropy.
    This bypasses all Keras loss function bugs.
    
    Args:
        y_true_masked: A stacked tensor from our data pipeline.
                       Shape: (batch_size, n_intervals, 2)
                       Channel 0: y_true (the labels)
                       Channel 1: y_mask (the mask)
        y_pred: The model's predictions.
                Shape: (batch_size, n_intervals)
    """
    # 1. Unpack the y_true and y_mask
    y_true = y_true_masked[..., 0]
    y_mask = y_true_masked[..., 1]
    
    # 2. Manually calculate binary cross-entropy
    # This formula is: -[y * log(p) + (1-y) * log(1-p)]
    
    # Clip y_pred to prevent log(0) or log(1) -> NaN
    y_pred_clipped = tf.clip_by_value(y_pred, EPSILON, 1. - EPSILON)
    
    # Calculate the per-interval loss
    loss_per_interval = -(
        y_true * tf.math.log(y_pred_clipped) +
        (1. - y_true) * tf.math.log(1. - y_pred_clipped)
    )
    
    # 3. Apply the mask manually
    masked_loss = loss_per_interval * y_mask
    
    # 4. Return the final, single-number loss
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(y_mask)


# --- Model Compile Function ---

def compile_ecg_survival_model(
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Convenience function: build and compile the model."""
    model = build_ecg_survival_model(
        n_intervals=N_INTERVALS,
        input_length=INPUT_LENGTH,
        n_leads=N_LEADS,
        base_filters=BASE_FILTERS,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # We compile with our NEW manual loss function
    model.compile(optimizer=optimizer, loss=survival_loss_with_mask)
    
    print(f"Model compiled with Adam (lr={learning_rate}) and survival_loss_with_mask.")
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print a summary
    # This code only runs if you execute `python3 model_architecture.py` directly
    print("--- Running model_architecture.py sanity check ---")
    model = build_ecg_survival_model()
    model.summary()
    print("Sanity check passed.")
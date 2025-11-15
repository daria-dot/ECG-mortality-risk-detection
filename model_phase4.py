"""\
Phase 4: Model Development for Discrete-Time Survival Prediction

This module defines a 1D-CNN model in Keras/TensorFlow to predict
10-year mortality using the discrete-time survival setup from Phase 3.

Key points (matching the project plan):
- Input: single ECG per example, shape (4096, 8)
  (4096 samples, 8 selected leads)
- Output: N_INTERVALS units (e.g., 120), each with sigmoid activation
  representing the conditional probability of surviving that interval.
- Loss: binary cross-entropy, applied per interval, with the Phase 3
  mask passed as `sample_weight` in model.fit.

The architecture is a reasonably deep 1D-CNN with residual blocks
(ResNet-style), but kept simple enough to run on modest hardware.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_bn_relu(
    x: keras.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
    name: str | None = None,
) -> keras.Tensor:
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
    x: keras.Tensor,
    filters: int,
    kernel_size: int = 7,
    strides: int = 1,
    name: str | None = None,
) -> keras.Tensor:
    """A simple ResNet-style residual block for 1D signals.

    Structure:
        - Conv1D -> BN -> ReLU
        - Conv1D -> BN
        - (Optional) 1x1 Conv on shortcut if shape mismatch
        - Add shortcut
        - ReLU
    """
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


def build_ecg_survival_model(
    n_intervals: int = 120,
    input_length: int = 4096,
    n_leads: int = 8,
    base_filters: int = 64,
) -> keras.Model:
    """Build the 1D-CNN model for ECG-based survival prediction.

    Args:
        n_intervals: Number of discrete time intervals (e.g., 120 months).
        input_length: Number of time samples per ECG (default: 4096).
        n_leads: Number of ECG leads after Phase 2 selection (default: 8).
        base_filters: Number of convolutional filters in the first block.

    Returns:
        A compiled Keras model (without specifying optimizer/loss).
        Use binary cross-entropy with sample_weight for training.
    """
    inputs = keras.Input(shape=(input_length, n_leads), name="ecg_input")

    # Initial conv + max-pooling (like a small "stem")
    x = conv_bn_relu(inputs, filters=base_filters, kernel_size=7, strides=2, name="stem")
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding="same", name="stem_pool")(x)

    # Residual stages: gradually increase filters and downsample
    # Stage 1: keep filters = base_filters
    x = residual_block(x, filters=base_filters, kernel_size=7, strides=1, name="stage1_block1")
    x = residual_block(x, filters=base_filters, kernel_size=7, strides=1, name="stage1_block2")

    # Stage 2: double filters, downsample
    x = residual_block(
        x,
        filters=base_filters * 2,
        kernel_size=7,
        strides=2,
        name="stage2_block1",
    )
    x = residual_block(
        x,
        filters=base_filters * 2,
        kernel_size=7,
        strides=1,
        name="stage2_block2",
    )

    # Stage 3: double filters again, downsample
    x = residual_block(
        x,
        filters=base_filters * 4,
        kernel_size=7,
        strides=2,
        name="stage3_block1",
    )
    x = residual_block(
        x,
        filters=base_filters * 4,
        kernel_size=7,
        strides=1,
        name="stage3_block2",
    )

    # Global average pooling over time
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Output: N_INTERVALS sigmoid units, one per time interval
    outputs = layers.Dense(
        n_intervals,
        activation="sigmoid",
        name="survival_logits",  # actually probabilities due to sigmoid
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ecg_survival_cnn")
    return model


def survival_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Binary cross-entropy per interval.

    This loss assumes:
      - y_true has shape (batch_size, n_intervals), with 1 = survived, 0 = died.
      - y_pred has shape (batch_size, n_intervals), outputs of a sigmoid.
      - The interval mask (from Phase 3) is passed via `sample_weight` in model.fit.

    In Keras you typically use:
        model.compile(optimizer="adam", loss=survival_loss)
        model.fit(X_train, y_true, sample_weight=y_mask, ...)
    """
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def compile_ecg_survival_model(
    n_intervals: int = 120,
    input_length: int = 4096,
    n_leads: int = 8,
    base_filters: int = 64,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Convenience function: build and compile the model.

    Uses Adam optimizer and binary cross-entropy loss.
    The Phase 3 mask should be passed as `sample_weight` to model.fit.
    """
    model = build_ecg_survival_model(
        n_intervals=n_intervals,
        input_length=input_length,
        n_leads=n_leads,
        base_filters=base_filters,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=survival_loss)
    return model


if __name__ == "__main__":
    # Quick sanity check: build the model and print a summary
    model = build_ecg_survival_model(n_intervals=120, input_length=4096, n_leads=8)
    model.summary()

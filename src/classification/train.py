"""GoogLeNet (Inception v1) model architecture, compilation, and training module.

Implements the authentic GoogLeNet architecture (Szegedy et al., 2014) using TensorFlow/Keras,
including parallel Inception multi-scale convolutional modules, optional auxiliary classifiers,
standardized training callbacks (checkpointing, early stopping, LR reduction), and learning curve visualization.
"""

from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

# ── Architecture Constants ───────────────────────────────────────────────────
DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_NUM_CLASSES = 4
RANDOM_STATE = 42


def _conv_block(x, filters, kernel_size, strides=(1, 1), padding="same", name=None):
    """Convolution + Batch Normalization + ReLU block."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=f"{name}_conv" if name else None,
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def _inception_module(
    x,
    f_1x1,
    f_3x3_reduce,
    f_3x3,
    f_5x5_reduce,
    f_5x5,
    f_pool_proj,
    name=None,
):
    """Inception v1 module with 4 parallel branches.

    Branch 1: 1x1 Convolution
    Branch 2: 1x1 Convolution (dim reduction) -> 3x3 Convolution
    Branch 3: 1x1 Convolution (dim reduction) -> 5x5 Convolution (two 3x3s)
    Branch 4: 3x3 Max Pooling -> 1x1 Convolution (projection)
    """
    # Branch 1: 1x1 conv
    b1 = _conv_block(x, f_1x1, (1, 1), name=f"{name}_b1")

    # Branch 2: 1x1 reduce -> 3x3 conv
    b2 = _conv_block(x, f_3x3_reduce, (1, 1), name=f"{name}_b2_reduce")
    b2 = _conv_block(b2, f_3x3, (3, 3), name=f"{name}_b2_3x3")

    # Branch 3: 1x1 reduce -> 5x5 conv (using two 3x3 for computational efficiency)
    b3 = _conv_block(x, f_5x5_reduce, (1, 1), name=f"{name}_b3_reduce")
    b3 = _conv_block(b3, f_5x5, (3, 3), name=f"{name}_b3_3x3_1")
    b3 = _conv_block(b3, f_5x5, (3, 3), name=f"{name}_b3_3x3_2")

    # Branch 4: 3x3 Max Pool -> 1x1 projection
    b4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=f"{name}_b4_pool")(x)
    b4 = _conv_block(b4, f_pool_proj, (1, 1), name=f"{name}_b4_proj")

    return layers.concatenate([b1, b2, b3, b4], axis=-1, name=name)


def _auxiliary_classifier(x, num_classes, name):
    """Auxiliary classifier for intermediate regularization in GoogLeNet."""
    x = layers.AveragePooling2D((5, 5), strides=(3, 3), padding="valid", name=f"{name}_avgpool")(x)
    x = _conv_block(x, 128, (1, 1), name=f"{name}_conv")
    x = layers.Flatten(name=f"{name}_flatten")(x)
    x = layers.Dense(1024, activation="relu", name=f"{name}_dense1")(x)
    x = layers.Dropout(0.7, seed=RANDOM_STATE, name=f"{name}_dropout")(x)
    x = layers.Dense(num_classes, activation="softmax", name=name)(x)
    return x


def build_googlenet(
    input_shape=DEFAULT_INPUT_SHAPE,
    num_classes=DEFAULT_NUM_CLASSES,
    include_auxiliary=False,
    dropout_rate=0.4,
):
    """Build the GoogLeNet (Inception v1) neural network architecture.

    Parameters
    ----------
    input_shape : tuple, default=(224, 224, 3)
        Input dimensions for the images.
    num_classes : int, default=4
        Number of output classification categories.
    include_auxiliary : bool, default=False
        Whether to output auxiliary classifiers (Inception 4a and 4d).
    dropout_rate : float, default=0.4
        Dropout rate before the final classification head.

    Returns
    -------
    tf.keras.Model
        The uncompiled Keras GoogLeNet model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Stage 1: Stem ────────────────────────────────────────────────────────
    # Conv 7x7/2 -> MaxPool 3x3/2 -> Conv 1x1 -> Conv 3x3 -> MaxPool 3x3/2
    x = _conv_block(inputs, 64, (7, 7), strides=(2, 2), name="stem_conv1")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="stem_maxpool1")(x)

    x = _conv_block(x, 64, (1, 1), name="stem_conv2_reduce")
    x = _conv_block(x, 192, (3, 3), name="stem_conv2")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="stem_maxpool2")(x)

    # ── Stage 2: Inception 3a, 3b ────────────────────────────────────────────
    x = _inception_module(x, 64, 96, 128, 16, 32, 32, name="inception_3a")
    x = _inception_module(x, 128, 128, 192, 32, 96, 64, name="inception_3b")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="maxpool_3")(x)

    # ── Stage 3: Inception 4a through 4e ─────────────────────────────────────
    x = _inception_module(x, 192, 96, 208, 16, 48, 64, name="inception_4a")
    aux1 = _auxiliary_classifier(x, num_classes, name="auxiliary_1") if include_auxiliary else None

    x = _inception_module(x, 160, 112, 224, 24, 64, 64, name="inception_4b")
    x = _inception_module(x, 128, 128, 256, 24, 64, 64, name="inception_4c")
    x = _inception_module(x, 112, 144, 288, 32, 64, 64, name="inception_4d")
    aux2 = _auxiliary_classifier(x, num_classes, name="auxiliary_2") if include_auxiliary else None

    x = _inception_module(x, 256, 160, 320, 32, 128, 128, name="inception_4e")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="maxpool_4")(x)

    # ── Stage 4: Inception 5a, 5b ────────────────────────────────────────────
    x = _inception_module(x, 256, 160, 320, 32, 128, 128, name="inception_5a")
    x = _inception_module(x, 384, 192, 384, 48, 128, 128, name="inception_5b")

    # ── Stage 5: Classification Head ────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, seed=RANDOM_STATE, name="classifier_dropout")(x)
    main_output = layers.Dense(num_classes, activation="softmax", name="main_output")(x)

    if include_auxiliary:
        outputs = [main_output, aux1, aux2]
    else:
        outputs = main_output

    return models.Model(inputs=inputs, outputs=outputs, name="GoogLeNet_InceptionV1")


def compile_googlenet(model, learning_rate=1e-4, include_auxiliary=False):
    """Compile the GoogLeNet model with Adam optimizer and sparse cross-entropy loss."""
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    if include_auxiliary:
        loss = {
            "main_output": losses.SparseCategoricalCrossentropy(),
            "auxiliary_1": losses.SparseCategoricalCrossentropy(),
            "auxiliary_2": losses.SparseCategoricalCrossentropy(),
        }
        loss_weights = {"main_output": 1.0, "auxiliary_1": 0.3, "auxiliary_2": 0.3}
        metrics = {"main_output": ["accuracy"], "auxiliary_1": ["accuracy"], "auxiliary_2": ["accuracy"]}
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    else:
        model.compile(
            optimizer=optimizer,
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    return model


def get_training_callbacks(
    checkpoint_path: Path,
    patience_early_stopping: int = 10,
    patience_reduce_lr: int = 4,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
):
    """Create standardized Keras callbacks for checkpointing, early stopping, and LR scheduling."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience_early_stopping,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=reduce_lr_factor,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1,
        ),
    ]
    return cbs


def plot_training_history(history, save_path=None):
    """Display training and validation loss and accuracy learning curves."""
    epochs = range(1, len(history.history["loss"]) + 1)
    acc_key = "accuracy" if "accuracy" in history.history else "main_output_accuracy"
    val_acc_key = f"val_{acc_key}"
    loss_key = "loss" if "loss" in history.history else "main_output_loss"
    val_loss_key = f"val_{loss_key}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs, history.history[acc_key], "b-o", label="Training Accuracy", markersize=4)
    ax1.plot(epochs, history.history[val_acc_key], "r-s", label="Validation Accuracy", markersize=4)
    ax1.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss plot
    ax2.plot(epochs, history.history[loss_key], "b-o", label="Training Loss", markersize=4)
    ax2.plot(epochs, history.history[val_loss_key], "r-s", label="Validation Loss", markersize=4)
    ax2.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curves saved to: {save_path}")
    plt.show()
    return fig


def train_model(
    train_ds,
    val_ds,
    model_dir: Path,
    epochs: int = 35,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.4,
    input_shape=DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
):
    """End-to-end training pipeline for GoogLeNet."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "googlenet_brain_tumor_best.keras"

    model = build_googlenet(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    model = compile_googlenet(model, learning_rate=learning_rate)
    training_callbacks = get_training_callbacks(checkpoint_path=checkpoint_path)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=1,
    )
    duration = time.time() - start_time
    print(f"Training completed in {duration / 60:.2f} minutes.")

    curves_path = model_dir / "learning_curves.png"
    plot_training_history(history, save_path=curves_path)

    final_model_path = model_dir / "googlenet_brain_tumor_final.keras"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}")

    return model, history

from pathlib import Path

import tensorflow as tf

from model import create_model, compile_model


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
INITIAL_LR = 1e-4
EPOCHS = 30
NUM_RUNS = 10

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir: str):
    train_dir = Path(data_dir) / "train"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, train_ds.class_names


def train_run(model, train_ds, val_ds, epochs: int = EPOCHS):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=0
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )
    return history

from pathlib import Path

import tensorflow as tf
import numpy as np

from model import create_model_v4, create_model_v5, compile_model_v4, compile_model_v5


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir: str, batch_size: int = BATCH_SIZE):
    train_dir = Path(data_dir) / "train"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
    )
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, train_ds.class_names


def train_run_v4(model, train_ds, val_ds, epochs: int = 30):
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=0,
    )
    return history


def train_run_v5(model, train_ds, val_ds, epochs: int = 100):
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model_run5.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=0,
    )
    return history


def ensemble_predict(models_list, val_ds):
    all_preds = []
    for model in models_list:
        preds = model.predict(val_ds, verbose=0)
        all_preds.append(preds)
    avg_preds = np.mean(all_preds, axis=0)
    return np.argmax(avg_preds, axis=-1)

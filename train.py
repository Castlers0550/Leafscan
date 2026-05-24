"""
Plant Disease Detection - Training Script
==========================================
RUN:
  py -3.11 train.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 15
SAVE_DIR    = "saved_model"
MODEL_PATH  = os.path.join(SAVE_DIR, "plant_disease_model.h5")
LABELS_PATH = os.path.join(SAVE_DIR, "class_names.txt")
TRAIN_DIR   = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"


def load_dataset():
    print("📦 Loading dataset from local folder...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42,
        validation_split=0.2,
        subset="validation",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ {num_classes} classes found!")
    print(f"   Sample classes: {class_names[:5]}...")

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    train_ds = (
        train_ds
        .map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, num_classes, class_names


def build_model(num_classes):
    print("🏗️  Building EfficientNetB0 model...")

    base = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"✅ Model built — {num_classes} output classes")
    return model, base


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_ds, val_ds, num_classes, class_names = load_dataset()
    model, base = build_model(num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # Phase 1 — train top layers only
    print("\n🚀 Phase 1: Training top layers...")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

    # Phase 2 — unfreeze and fine-tune
    print("\n🔧 Phase 2: Fine-tuning entire model...")
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=len(history1.epoch) + 10,
        initial_epoch=len(history1.epoch),
        callbacks=cb,
    )

    # Save class names
    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(class_names))

    print(f"\n✅ Model saved  → {MODEL_PATH}")
    print(f"✅ Labels saved → {LABELS_PATH}")

    # Plot accuracy
    acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    plt.figure(figsize=(8, 4))
    plt.plot(acc, marker='o')
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_history.png"))
    print("📊 Plot saved")

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n🎯 Final validation accuracy: {acc*100:.1f}%")


if __name__ == "__main__":
    main()
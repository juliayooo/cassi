"""Combine the image model with facial landmark input (experimental).

Takes the trained image model, adds a second input branch for facial
landmarks (from extract_landmarks.py), merges both, and fine-tunes the
result. This was an exploration and is not used by the live scripts.
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model

# ---- Configuration -------------------------------------------------------
BASE_MODEL = "../../models/trained_model3.h5"
DATA_DIR = "split-human-faces-resized"
OUTPUT_MODEL = "trained_model_landmarks.h5"
LANDMARK_SHAPE = (10, 2)  # (points, xy) fed to the landmark branch
BATCH_SIZE = 32
EPOCHS = 10
# --------------------------------------------------------------------------


def create_dataset(image_dir, landmark_file, batch_size=BATCH_SIZE):
    """Load images and their landmarks, padding landmarks to LANDMARK_SHAPE."""
    data = np.load(landmark_file, allow_pickle=False)

    images = []
    landmarks_list = []
    labels = []

    for img_name in data.files:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.resize(cv2.imread(img_path), (224, 224)) / 255.0
        images.append(img)

        landmarks = data[img_name]
        if isinstance(landmarks, dict):
            landmarks = np.array(list(landmarks.values()))
        else:
            landmarks = np.array(landmarks)

        if landmarks.shape != LANDMARK_SHAPE:
            padded = np.zeros(LANDMARK_SHAPE)
            n = min(LANDMARK_SHAPE[0], landmarks.shape[0])
            padded[:n] = landmarks[:n]
            landmarks = padded
        landmarks_list.append(landmarks)

        labels.append(1 if "yes" in image_dir.lower() else 0)

    dataset = tf.data.Dataset.from_tensor_slices((
        {"input_1": np.array(images), "landmarks": np.array(landmarks_list)},
        np.array(labels),
    ))
    return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


model = load_model(BASE_MODEL)
model.summary()

# Landmark branch
landmark_input = layers.Input(shape=LANDMARK_SHAPE, name="landmarks")
x = layers.Flatten(name="landmark_input")(landmark_input)
x = layers.Dense(32, activation="relu", name="landmark_dense_1")(x)

# Merge with the image model's output
merged = layers.concatenate([model.output, x], name="merged_layers")
merged = layers.Dense(64, activation="relu", name="merged_dense_1")(merged)
output = layers.Dense(1, activation="sigmoid", name="output")(merged)

updated_model = Model(inputs=[model.input, landmark_input], outputs=output)
updated_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
updated_model.summary()

train_dataset = create_dataset(
    os.path.join(DATA_DIR, "train"),
    os.path.join(DATA_DIR, "train", "train_face_landmarks.npz"),
)
val_dataset = create_dataset(
    os.path.join(DATA_DIR, "val"),
    os.path.join(DATA_DIR, "val", "val_face_landmarks.npz"),
)

updated_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

updated_model.save(OUTPUT_MODEL)

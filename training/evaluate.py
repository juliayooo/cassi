"""Visually check a model's predictions.

Loads one batch from DATA_DIR (yes/ and no/ subfolders) and shows a 5x5 grid
of images titled with the predicted and real labels.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ---- Configuration -------------------------------------------------------
MODEL_PATH = "../models/trained_model3.h5"
DATA_DIR = "data2/test"
GRID = 5  # Shows GRID x GRID images
# --------------------------------------------------------------------------

model = load_model(MODEL_PATH)
model.summary()
print("Available GPUs:", len(tf.config.list_physical_devices("GPU")))

pred_dataset = image_dataset_from_directory(
    DATA_DIR,
    image_size=(224, 224),
    batch_size=GRID * GRID,
    labels="inferred",
)
class_names = pred_dataset.class_names

for images, labels in pred_dataset.take(1):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(min(GRID * GRID, len(images))):
        plt.subplot(GRID, GRID, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(
            f"Pred: {class_names[predicted_labels[i]]} | "
            f"Real: {class_names[labels[i].numpy()]}"
        )
        plt.axis("off")
    plt.show()

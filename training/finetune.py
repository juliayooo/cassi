"""Fine-tune an existing model on a new yes/no dataset.

Expects DATA_DIR to contain train/ and val/ folders, each with yes/ and no/
subfolders of 224x224 images.
"""
import os

import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ---- Configuration -------------------------------------------------------
BASE_MODEL = "../models/trained_model3.h5"
DATA_DIR = "set-images-cropped-resized"
OUTPUT_MODEL = "trained_model6.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 20
# --------------------------------------------------------------------------

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

model = load_model(BASE_MODEL)
model.summary()
print("Available GPUs:", len(tf.config.list_physical_devices("GPU")))


def remove_corrupt_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Removing corrupt image: {file_path} - Error: {e}")
                os.remove(file_path)


remove_corrupt_images(train_dir)

train_dataset = image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, labels="inferred"
)
val_dataset = image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, labels="inferred"
)
print("Classes:", train_dataset.class_names)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

model.save(OUTPUT_MODEL)

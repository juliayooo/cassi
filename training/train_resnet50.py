"""Train the base yes/no face classifier.

Transfer learning on ResNet50 (ImageNet weights, frozen) with a small
classification head. Expects DATA_DIR to contain train/, val/ and test/
folders, each with yes/ and no/ subfolders of 224x224 images.
"""
import os

import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ---- Configuration -------------------------------------------------------
DATA_DIR = "resized_dataset"
OUTPUT_MODEL = "trained_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 20
# --------------------------------------------------------------------------

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

print("Available GPUs:", len(tf.config.list_physical_devices("GPU")))
print("TensorFlow", tf.__version__)


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

class_names = train_dataset.class_names
print("Classes:", class_names)

# Pretrained ResNet50 base, frozen so our data doesn't change its weights
pt_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
pt_model.trainable = False

head = pt_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(256, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(len(class_names), activation="softmax")(head)

model = Model(inputs=pt_model.input, outputs=head)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

model.save(OUTPUT_MODEL)

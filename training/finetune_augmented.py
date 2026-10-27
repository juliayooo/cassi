"""Fine-tune an existing model with data augmentation.

Applies random fog and random resized crops (albumentations) to every
training batch, shows a preview of 10 augmented images, then trains.
Expects DATA_DIR to contain train/ and val/ folders, each with yes/ and no/
subfolders of 224x224 images.
"""
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ---- Configuration -------------------------------------------------------
BASE_MODEL = "trained_model4.h5"
DATA_DIR = "full-split-human-faces-resized"
OUTPUT_MODEL = "trained_model_postaug.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SHOW_PREVIEW = True  # Show 10 augmented images before training
# --------------------------------------------------------------------------

transform = A.Compose([
    A.RandomFog(p=1.0),
    A.RandomResizedCrop(size=IMG_SIZE, scale=(0.85, 1.0), ratio=(0.8, 1.3), p=0.9),
])


def augment(batch_images, label):
    """Augment each image in a (batch, 224, 224, 3) batch individually."""
    batch_images = batch_images.numpy().astype(np.uint8)

    augmented_images = []
    for i, img in enumerate(batch_images):
        img = np.ascontiguousarray(img)  # Helps OpenCV functions

        if img.shape[0] != IMG_SIZE[0] or img.shape[1] != IMG_SIZE[1]:
            print(f"Warning: Image {i} shape is {img.shape}. Resizing.")
            img = cv2.resize(img, IMG_SIZE)

        try:
            augmented_img = transform(image=img)["image"]
        except Exception as e:
            print(f"Error augmenting image {i}: {e}")
            augmented_img = img  # Fall back to the original image

        augmented_images.append(augmented_img)

    # Normalize back to [0, 1]
    augmented_images = np.stack(augmented_images).astype(np.float32) / 255.0
    return tf.convert_to_tensor(augmented_images), label


def tf_augment_fn(images, labels):
    aug_images, labels = tf.py_function(
        func=augment, inp=[images, labels], Tout=[tf.float32, tf.int32]
    )
    aug_images.set_shape((None, *IMG_SIZE, 3))
    labels.set_shape((None,))
    return aug_images, labels


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


train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

remove_corrupt_images(train_dir)

train_dataset = image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, labels="inferred"
)
val_dataset = image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, labels="inferred"
)

# Datasets are already batched, so map the augmentation per batch
train_dataset = train_dataset.map(tf_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

model = load_model(BASE_MODEL)
model.summary()
print("Available GPUs:", len(tf.config.list_physical_devices("GPU")))

model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

if SHOW_PREVIEW:
    for batch_images, batch_labels in train_dataset.take(1):
        images_np = batch_images.numpy()
        print("Augmented batch shape:", images_np.shape)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(images_np[i])
            ax.set_title(f"Aug {i + 1}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()  # Close the window to start training

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

model.save(OUTPUT_MODEL)

"""Extract facial landmarks with the Google Cloud Vision API (experimental).

Scans every image under DATASET_DIR, detects face landmarks, and merges the
results into an .npz file keyed by image filename. Requires Google Cloud
credentials (GOOGLE_APPLICATION_CREDENTIALS).
"""
import os

import numpy as np
from google.cloud import vision

# ---- Configuration -------------------------------------------------------
DATASET_DIR = "split-human-faces-resized/train/no"
NPZ_FILE = "split-human-faces-resized/train/train_face_landmarks.npz"
BATCH_SIZE = 200
# --------------------------------------------------------------------------


def save_landmarks(new_data, output_file=NPZ_FILE):
    """Merge new landmark data into the existing .npz file."""
    if os.path.exists(output_file):
        existing_data = dict(np.load(output_file, allow_pickle=True))
    else:
        existing_data = {}

    existing_data.update(new_data)
    np.savez_compressed(output_file, **existing_data)
    print(f"Updated {output_file} with {len(new_data)} new entries.")


def detect_faces(image_path):
    """Return {face_i: {landmark_type: (x, y)}} for an image."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    response = client.face_detection(image=vision.Image(content=content))
    if response.error.message:
        raise Exception(
            f"{response.error.message}\n"
            "For more info, check: https://cloud.google.com/apis/design/errors"
        )

    landmarks_dict = {}
    for i, face in enumerate(response.face_annotations):
        landmarks_dict[f"face_{i}"] = {
            landmark.type_: (landmark.position.x, landmark.position.y)
            for landmark in face.landmarks
        }
    return landmarks_dict


def process_directory(base_dir, output_file=NPZ_FILE, batch_size=BATCH_SIZE):
    """Detect landmarks for all images in base_dir, saving after each batch."""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, file))

    for i in range(0, len(image_paths), batch_size):
        image_landmarks = {}
        for image_path in image_paths[i : i + batch_size]:
            try:
                landmarks = detect_faces(image_path)
                if landmarks:
                    image_landmarks[os.path.basename(image_path)] = landmarks
                print(f"Processed: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        save_landmarks(image_landmarks, output_file)


if __name__ == "__main__":
    process_directory(DATASET_DIR)

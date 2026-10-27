"""Hands-free live classifier (no keyboard or display needed).

Every few seconds, grabs a webcam frame, classifies any detected face as
"yes" or "no", and signals the Arduino. Stops after MAX_PREDICTIONS
predictions. Intended for unattended runs, e.g. via deploy/cassi.service.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import serial
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

# ---- Configuration -------------------------------------------------------
MODEL_PATH = ROOT / "models" / "trained_model3.h5"
CASCADE_PATH = ROOT / "assets" / "haarcascade_frontalface_alt.xml"
SERIAL_PORT = "/dev/tty.usbmodem101"  # Arduino port (check with `ls /dev/tty.*`)
BAUD_RATE = 9600
CAMERA_INDEX = 1
MARGIN = 30  # Pixels added around the detected face box
INTERVAL_SECONDS = 5  # Time between captures
MAX_PREDICTIONS = 15
CLASS_NAMES = ["no", "yes"]
# --------------------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)
print("Available GPUs:", len(tf.config.list_physical_devices("GPU")))

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
cap = cv2.VideoCapture(CAMERA_INDEX)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
except serial.SerialException:
    ser = None
    print(f"No Arduino on {SERIAL_PORT}; signals will only be printed.")


def trigger_light(num):
    """Send a signal to the Arduino: 1 = yes, 2 = no."""
    if ser:
        ser.write(str(num).encode())
    else:
        print(f"[Arduino] would send {num}")


def classify(faceimg):
    """Return (class name, confidence) for a 224x224 BGR face crop.

    The model was trained on RGB images with 0-255 pixel values (as loaded by
    image_dataset_from_directory), so convert to match. Only the first
    len(CLASS_NAMES) outputs are real classes.
    """
    rgb = cv2.cvtColor(faceimg, cv2.COLOR_BGR2RGB).astype("float32")
    scores = model.predict(np.expand_dims(rgb, axis=0), verbose=0)[0][: len(CLASS_NAMES)]
    scores = scores / scores.sum()
    idx = int(np.argmax(scores))
    return CLASS_NAMES[idx], float(scores[idx])


preds = 0

print(f"{INTERVAL_SECONDS} seconds to prepare")

while preds < MAX_PREDICTIONS:
    time.sleep(INTERVAL_SECONDS)
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        preds += 1

        # Expand bounding box slightly
        x1 = max(0, x - MARGIN)
        y1 = max(0, y - MARGIN)
        x2 = min(img.shape[1], x + w + MARGIN)
        y2 = min(img.shape[0], y + h + MARGIN)

        faceimg = cv2.resize(img[y1:y2, x1:x2], (224, 224))
        predicted_class, confidence = classify(faceimg)
        print(f"Prediction: {predicted_class} ({confidence:.0%})")

        if predicted_class == "yes":
            trigger_light(1)
        elif predicted_class == "no":
            trigger_light(2)

cap.release()

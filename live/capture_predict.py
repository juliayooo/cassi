"""Interactive live classifier.

Shows the webcam feed with detected faces outlined. Press SPACE to capture
the detected face, classify it as "yes" or "no", save the crop into
captures/<class>/, and signal the Arduino. Press ESC to quit.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import serial
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

# ---- Configuration -------------------------------------------------------
MODEL_PATH = ROOT / "models" / "trained_model3.h5"
CASCADE_PATH = ROOT / "assets" / "haarcascade_frontalface_alt.xml"
CAPTURE_DIR = ROOT / "captures"
SERIAL_PORT = "/dev/tty.usbmodem101"  # Arduino port (check with `ls /dev/tty.*`)
BAUD_RATE = 9600
CAMERA_INDEX = 2  # 0 = iPhone via OBS, 2 = webcam via OBS
MARGIN = 30  # Pixels added around the detected face box
CLASS_NAMES = ["no", "yes"]
# --------------------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)
print("Available GPUs:", len(tf.config.list_physical_devices("GPU")))

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


for name in CLASS_NAMES:
    os.makedirs(CAPTURE_DIR / name, exist_ok=True)

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
cap = cv2.VideoCapture(CAMERA_INDEX)
counter = 0
status = "Press SPACE to classify, ESC or Q to quit"

# Bring the window to the front so it receives key presses
WINDOW = "Face Capture"
cv2.namedWindow(WINDOW)
cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)

print("Click the video window, then press SPACE to capture a face, or ESC/Q to exit.")

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    display = img.copy()  # Draw on a copy so saved crops stay clean
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(WINDOW, display)
    key = cv2.waitKey(1) & 0xFF

    # Quit on ESC, Q, or closing the window
    if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == 32 and len(faces) == 0:  # SPACE with no face in view
        status = "No face detected - try again"
        print(status)

    if key == 32 and len(faces) > 0:  # SPACE
        for (x, y, w, h) in faces:
            counter += 1

            # Expand bounding box slightly
            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(img.shape[1], x + w + MARGIN)
            y2 = min(img.shape[0], y + h + MARGIN)

            faceimg = cv2.resize(img[y1:y2, x1:x2], (224, 224))
            predicted_class, confidence = classify(faceimg)

            save_path = CAPTURE_DIR / predicted_class / f"face_{counter}.png"
            cv2.imwrite(str(save_path), faceimg)
            print(f"Face saved: {save_path} | Predicted as: {predicted_class} ({confidence:.0%})")
            status = f"Face {counter}: {predicted_class} ({confidence:.0%})"

            if predicted_class == "yes":
                trigger_light(1)
            elif predicted_class == "no":
                trigger_light(2)

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # macOS needs this to actually close the window

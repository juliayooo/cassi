# CASSI

A live face classifier connected to an Arduino. A webcam watches for faces. Each detected face goes to a trained image model that labels it **yes** or **no**, and the result is sent over serial to an Arduino, which lights up.

```
webcam ──► face detection ──► ResNet50 classifier ──► "yes" / "no" ──► Arduino (serial)
          (OpenCV Haar cascade)   (models/trained_model3.h5)            1 = yes, 2 = no
```

## Repository layout

```
CASSI/
├── live/                     Run the installation
│   ├── capture_predict.py    Interactive: press SPACE to classify, ESC to quit
│   ├── auto_capture.py       Hands-free: classifies a face every few seconds
│   └── arduino_test.py       Sends one test signal to the Arduino
├── training/                 Build and check the model
│   ├── train_resnet50.py     Train the base classifier (ResNet50 transfer learning)
│   ├── finetune.py           Fine-tune an existing model on a new dataset
│   ├── finetune_augmented.py Fine-tune with fog / random-crop augmentation
│   ├── evaluate.py           Show a grid of predictions vs. real labels
│   └── experimental/         Facial-landmark model (Google Cloud Vision), not used live
├── data_prep/                Build the dataset
│   ├── download_faces.js     Download AI-generated faces (thispersondoesnotexist.com)
│   ├── label_tool.py         GUI for hand-labeling images as Yes / No
│   ├── crop.py               Crop images to the face region
│   ├── resize.py             Resize images to 224×224 PNG
│   ├── renumber_dataset.py   Copy a split dataset with uniform file names
│   └── remove_mac_files.py   Delete macOS "._" files that break loading
├── models/
│   └── trained_model3.h5     Trained model used by the live scripts
├── assets/
│   └── haarcascade_frontalface_alt.xml   OpenCV face detector
├── deploy/
│   ├── run.sh                Activates conda and starts auto_capture.py
│   └── cassi.service         systemd unit to run on boot
└── requirements.txt
```

## Setup

Requires Python 3.9–3.11.

```bash
git clone git@github.com:juliayooo/model.git && cd model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

1. Connect the Arduino and find its port:
   ```bash
   ls /dev/tty.usbmodem*
   ```
2. Open `live/capture_predict.py` and update the settings at the top:
   - `SERIAL_PORT`: the Arduino port from step 1
   - `CAMERA_INDEX`: which camera to use (`0`, `1`, `2`, …). With OBS, `0` was the iPhone and `2` the webcam.
3. Test the serial connection, then start the classifier:
   ```bash
   python live/arduino_test.py
   python live/capture_predict.py
   ```
   Click the video window so it has focus, then press **SPACE** to capture and classify the face on screen. Press **ESC** or **Q**, or close the window, to quit. Captured faces are saved to `captures/yes/` or `captures/no/`. That folder is git-ignored.

For an unattended setup with no keyboard or screen, use `live/auto_capture.py` instead. It classifies a face every 5 seconds, up to 15 times. These values are set by `INTERVAL_SECONDS` and `MAX_PREDICTIONS`. To start it on boot (Linux / Raspberry Pi), see `deploy/cassi.service`.

### Arduino protocol

The Arduino receives one ASCII character over serial at 9600 baud:

| Signal | Meaning          |
| ------ | ---------------- |
| `1`    | Face classified **yes** |
| `2`    | Face classified **no**  |

## Training your own model

All scripts in `training/` and `data_prep/` have a **Configuration** block at the top. Edit the paths there before you run a script.

1. **Collect images:** `cd data_prep && npm install && node download_faces.js`
2. **Label them:** `python data_prep/label_tool.py`
3. **Prepare them:** `crop.py` → `resize.py` → split into `train/`, `val/`, `test/` folders. Each of those needs `yes/` and `no/` subfolders.
4. **Train:** `python training/train_resnet50.py`
5. **Improve (optional):** `finetune.py` or `finetune_augmented.py`
6. **Check results:** `python training/evaluate.py`

Expected dataset structure:

```
dataset/
├── train/ ├── yes/ └── no/
├── val/   ├── yes/ └── no/
└── test/  ├── yes/ └── no/
```

Datasets are not included in this repo.

## Model

- **Architecture:** ResNet50 (ImageNet weights, frozen) → AveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(softmax)
- **Input:** 224×224 RGB face crop
- **Classes:** `no`, `yes` (in alphabetical folder order)
- **File:** `models/trained_model3.h5` (Keras HDF5, ~100 MB)

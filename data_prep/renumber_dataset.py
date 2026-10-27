"""Copy a split dataset into a new folder with uniform file names.

Reads INPUT_DIR/<split>/<class>/<image> (split = train/val/test, class =
yes/no) and writes OUTPUT_DIR/<split>/<class>/person_00000.png, ...
"""
import glob
import os

import cv2

# ---- Configuration -------------------------------------------------------
INPUT_DIR = "resized_augment2_2"
OUTPUT_DIR = "data2"
# --------------------------------------------------------------------------

for ct, file in enumerate(sorted(glob.glob(os.path.join(INPUT_DIR, "*", "*", "*")))):
    split = os.path.basename(os.path.dirname(os.path.dirname(file)))
    category = os.path.basename(os.path.dirname(file))

    out_dir = os.path.join(OUTPUT_DIR, split, category)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(file)
    cv2.imwrite(os.path.join(out_dir, f"person_{ct:05}.png"), img)

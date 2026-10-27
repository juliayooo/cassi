"""Simple GUI for hand-labeling images as Yes or No.

Click "Load Images" and pick a folder. Each image is shown one at a time;
clicking Yes or No copies it into OUTPUT_DIR/Yes or OUTPUT_DIR/No. When all
images are labeled, a labels.csv is written to OUTPUT_DIR.
"""
import os
import shutil
from tkinter import Button, Label, Tk, filedialog

import pandas as pd
from PIL import Image, ImageTk

# ---- Configuration -------------------------------------------------------
OUTPUT_DIR = "sorted_set_human"
# --------------------------------------------------------------------------

yes_dir = os.path.join(OUTPUT_DIR, "Yes")
no_dir = os.path.join(OUTPUT_DIR, "No")
metadata_file = os.path.join(OUTPUT_DIR, "labels.csv")

os.makedirs(yes_dir, exist_ok=True)
os.makedirs(no_dir, exist_ok=True)

current_image_index = 0
image_list = []
labels = []


def load_images():
    global image_list, current_image_index
    folder = filedialog.askdirectory(title="Select Image Folder")
    if folder:
        image_list = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        current_image_index = 0
        show_image()


def show_image():
    if current_image_index < len(image_list):
        img = Image.open(image_list[current_image_index])
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk
        window.title(f"Labeling {current_image_index + 1}/{len(image_list)}")
    else:
        save_metadata()
        window.title("Labeling Completed!")
        label.config(text="All images labeled!")


def label_image(label_value):
    global current_image_index
    if current_image_index < len(image_list):
        img_path = image_list[current_image_index]
        labels.append({"image": img_path, "label": label_value})

        dest_dir = yes_dir if label_value == "Yes" else no_dir
        shutil.copy(img_path, dest_dir)

        current_image_index += 1
        show_image()


def save_metadata():
    if labels:
        pd.DataFrame(labels).to_csv(metadata_file, index=False)
        print(f"Metadata saved to {metadata_file}")


window = Tk()
window.title("Dataset Creation Tool")
label = Label(window)
label.pack()

Button(window, text="Load Images", command=load_images).pack()
Button(window, text="Yes", command=lambda: label_image("Yes")).pack(side="left", padx=10)
Button(window, text="No", command=lambda: label_image("No")).pack(side="right", padx=10)

window.mainloop()

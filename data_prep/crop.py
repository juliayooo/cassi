"""Crop every image in INPUT_DIR/{yes,no} to a fixed box around the face."""
import os

from PIL import Image

# ---- Configuration -------------------------------------------------------
INPUT_DIR = "set_to_crop"
OUTPUT_DIR = "set-images-cropped"
CROP_BOX = (17, 37, 162, 182)  # (left, top, right, bottom)
# --------------------------------------------------------------------------


def crop_images(input_dir, output_dir, box):
    for category in ["yes", "no"]:
        input_category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        for filename in os.listdir(input_category_dir):
            if filename.startswith("."):
                continue
            input_path = os.path.join(input_category_dir, filename)
            output_path = os.path.join(output_category_dir, filename)

            try:
                with Image.open(input_path) as img:
                    img.crop(box).save(output_path)
                    print(f"Cropped and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    crop_images(INPUT_DIR, OUTPUT_DIR, CROP_BOX)

"""Resize every image in INPUT_DIR/{yes,no} to 224x224 and save as PNG."""
import os

from PIL import Image

# ---- Configuration -------------------------------------------------------
INPUT_DIR = "set-images-cropped/val"
OUTPUT_DIR = "set-images-cropped-resized/val"
TARGET_SIZE = (224, 224)  # (width, height)
# --------------------------------------------------------------------------


def resize_images(input_dir, output_dir, target_size):
    for category in ["yes", "no"]:
        input_category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        for filename in os.listdir(input_category_dir):
            if filename.startswith("."):
                continue
            input_path = os.path.join(input_category_dir, filename)
            output_path = os.path.join(
                output_category_dir, os.path.splitext(filename)[0] + ".png"
            )

            try:
                with Image.open(input_path) as img:
                    img.resize(target_size).save(output_path)
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    resize_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)

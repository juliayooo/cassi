"""Delete macOS "._" metadata files that break image loading."""
import os

# ---- Configuration -------------------------------------------------------
DIRECTORIES = [
    "resized_dataset/train",
    "resized_dataset/val",
    "resized_dataset/test",
]
# --------------------------------------------------------------------------


def remove_mac_hidden_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("._"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed hidden file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")


if __name__ == "__main__":
    for directory in DIRECTORIES:
        remove_mac_hidden_files(directory)

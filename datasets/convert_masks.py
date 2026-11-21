import os
import cv2
import numpy as np
import argparse
import shutil

# ===============================
# Argument parser
# ===============================
parser = argparse.ArgumentParser(
    description="Convert segmentation masks to YOLO polygon format and copy images"
)

parser.add_argument(
    "--root",
    type=str,
    required=True,
    help="Root folder containing segmentation_task directory"
)

parser.add_argument(
    "--output",
    type=str,
    default="YOLO",
    help="Output folder to store images and labels"
)

args = parser.parse_args()

root_folder = args.root
output_folder = args.output

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# ===============================
# Define dataset folders
# ===============================
splits = ["train", "test"]

# ===============================
# Class mapping based on filename
# ===============================
def get_class_id(filename):
    if "_gl_" in filename:
        return 0
    elif "_me_" in filename:
        return 1
    elif "_pi_" in filename:
        return 2
    else:
        return None


# ===============================
# Process dataset
# ===============================
for split in splits:

    mask_folder = os.path.join(root_folder, "segmentation_task", split, "masks")
    image_folder = os.path.join(root_folder, "segmentation_task", split, "images")

    print(f"Processing split: {split}")

    if not os.path.exists(mask_folder):
        print(f"Mask folder not found: {mask_folder}")
        continue

    for filename in os.listdir(mask_folder):

        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue

        class_id = get_class_id(filename)

        if class_id is None:
            print(f"Skipping (unknown class): {filename}")
            continue

        mask_path = os.path.join(mask_folder, filename)

        # Read mask as grayscale
        mask = cv2.imread(mask_path, 0)

        if mask is None:
            print(f"Failed to read mask: {filename}")
            continue

        # Convert mask to binary
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Extract contours
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        h, w = mask.shape

        base_name = os.path.splitext(filename)[0]

        txt_path = os.path.join(output_folder, base_name + ".txt")

        with open(txt_path, "w") as f:

            for contour in contours:

                # Skip very small contours
                if cv2.contourArea(contour) < 10:
                    continue

                # Simplify polygon
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) < 3:
                    continue

                segmentation = []

                # Normalize coordinates
                for point in approx:
                    x = point[0][0] / w
                    y = point[0][1] / h
                    segmentation.append(f"{x:.6f}")
                    segmentation.append(f"{y:.6f}")

                line = f"{class_id} " + " ".join(segmentation)
                f.write(line + "\n")

        # ===============================
        # Copy corresponding image
        # ===============================
        image_extensions = [".jpg", ".jpeg", ".png"]

        for ext in image_extensions:
            image_path = os.path.join(image_folder, base_name + ext)

            if os.path.exists(image_path):
                shutil.copy2(image_path, os.path.join(output_folder, base_name + ext))
                break

        print(f"Processed: {base_name}")

print("Conversion finished 🚀")
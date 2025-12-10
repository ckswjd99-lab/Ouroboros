import json
import cv2
import os


sequence = "bear"

json_path = f"/data/DAVIS/Annotations_bbox/480p/{sequence}.json"
image_folder = f"/data/DAVIS/JPEGImages/480p/{sequence}"
output_folder = f"./output/bbox/{sequence}"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Load JSON file containing bounding box data
with open(json_path, "r") as f:
    bbox_data = json.load(f)

# Iterate over each image in the JSON file
for img_name, objects in bbox_data.items():
    img_path = os.path.join(image_folder, f"{img_name}.jpg")
    
    if not os.path.exists(img_path):
        print(f"Image {img_name}.jpg not found, skipping.")
        continue

    # Read the image using OpenCV
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw each bounding box on the image
    for obj in objects:
        x_min, y_min, x_max, y_max = obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]
        label = obj["label"]

        color = (255, 0, 0)  # Red color for the bounding box
        thickness = 2

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image with bounding boxes (convert back to BGR for OpenCV)
    output_path = os.path.join(output_folder, f"{img_name}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

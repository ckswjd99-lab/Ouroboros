import os
import json
import numpy as np
import cv2

from tqdm import tqdm

input_base_path = "/data/DAVIS/Annotations/480p/"
output_base_path = "/data/DAVIS/Annotations_bbox/480p/"

os.makedirs(output_base_path, exist_ok=True)

sequences = sorted(os.listdir(input_base_path))

def extract_bounding_boxes(mask):
    unique_labels = np.unique(mask)
    bounding_boxes = []
    
    for label in unique_labels:
        if label == 0:
            continue
        
        y_indices, x_indices = np.where(mask == label)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        bounding_boxes.append({
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "label": f"{label}"
        })
    
    return bounding_boxes

for sequence in sequences:
    sequence_path = os.path.join(input_base_path, sequence)
    output_file = os.path.join(output_base_path, f"{sequence}.json")
    
    if not os.path.isdir(sequence_path):
        continue
    
    frames = sorted(os.listdir(sequence_path))
    sequence_data = {}

    pbar = tqdm(frames, desc=f"Processing {sequence}")
    for frame in tqdm(frames):
        if not frame.endswith(".png"):
            continue
        
        frame_path = os.path.join(sequence_path, frame)
        frame_number = os.path.splitext(frame)[0]
        
        mask = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        bounding_boxes = extract_bounding_boxes(mask)
        
        if bounding_boxes:
            sequence_data[frame_number] = bounding_boxes

    with open(output_file, "w") as f:
        json.dump(sequence_data, f, indent=4)

output_base_path

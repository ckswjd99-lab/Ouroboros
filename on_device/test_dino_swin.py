import torch
from torchvision import transforms
import cv2

from PIL import Image

from ipconv.models.DINO import build_dino_4scale_swin, build_dino_5scale_swin
from ipconv.models.DINO.util.misc import nested_tensor_from_tensor_list

import os

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# INPUT_PATH = '00000.jpg'
# OUTPUT_PATH = 'output.jpg'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_image(image_path):
    image_orig = Image.open(image_path).convert("RGB")
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # orig_image_size = torch.tensor(image_orig.size[::-1])
    orig_image_size = torch.tensor((1024, 1024), dtype=torch.float32)  # Assuming fixed size for padding

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
            normalize,
        ])
    image = transform(image_orig)
    image_padded = torch.zeros((3, 1024, 1024), dtype=image.dtype, device=image.device)
    center_shift_x = (1024 - image.shape[1]) // 2
    center_shift_y = (1024 - image.shape[2]) // 2
    image_padded[:, center_shift_x:center_shift_x + image.shape[1], center_shift_y:center_shift_y + image.shape[2]] = image
    image = image_padded

    image_orig = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    image_t = transform(image_orig)
    image_t_padded = torch.zeros((3, 1024, 1024), dtype=image_t.dtype, device=image_t.device)
    center_shift_x = (1024 - image_t.shape[1]) // 2
    center_shift_y = (1024 - image_t.shape[2]) // 2
    image_t_padded[:, center_shift_x:center_shift_x + image_t.shape[1], center_shift_y:center_shift_y + image_t.shape[2]] = image_t
    image_t = image_t_padded

    return image, orig_image_size, image_t

def visualize_detections(image, boxes, labels, scores, conf_thresh, output_path):
    # draw box and label on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{COCO_CLASSES[label]}: {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(output_path, image)


def main():
    # model, _, postprocessors = build_dino_4scale_swin()
    model, _, postprocessors = build_dino_5scale_swin()
    model.eval()
    model = model.to(DEVICE)

    # load images from the input directory
    input_dir = "/data/DAVIS/JPEGImages/480p/flamingo"
    image_paths = os.listdir(input_dir)

    output_dir = "./output/test_dino_swin"
    os.makedirs(output_dir, exist_ok=True)

    for image_path in sorted(image_paths):
        image_path = os.path.join(input_dir, image_path)

        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_path}")
            continue

        image, orig_image_size, image_t = preprocess_image(image_path)
        image = image.to(DEVICE)
        orig_image_size = orig_image_size.to(DEVICE)
        print(f"Image size: {orig_image_size}")

        images = nested_tensor_from_tensor_list([image])
        orig_image_sizes = torch.stack([orig_image_size])

        # forward
        with torch.no_grad():
            import time

            # for _ in range(10):
            #     outputs = model(images)

            num_repeats = 1
            start_time = time.time()
            for _ in range(num_repeats):
                outputs = model(images)
            end_time = time.time()
            print(f"Inference time for num_repeats iterations: {(end_time - start_time) / num_repeats:.4f} seconds per iteration")

        # postprocess
        predictions = postprocessors['bbox'](outputs, orig_image_sizes)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # boxes = (boxes * 1024).astype(int)  # Scale boxes to original image size


        # original_image = cv2.imread(INPUT_PATH)
        original_image = image_t.permute(1, 2, 0).cpu().numpy()
        image_basename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"output_{image_basename}")
        visualize_detections(
            original_image,
            boxes,
            labels,
            scores,
            0.5,
            output_path
            )
        print(f"Processed {image_path}, saved output to {output_path}")


if __name__ == "__main__":
    main()
    print(f"Output saved")
    print("Validation completed successfully.")
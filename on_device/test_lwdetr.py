import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from ipconv.models.LW_DETR import build_lwdetr_small, build_lwdetr_xlarge
from ipconv.models.LW_DETR.util.misc import nested_tensor_from_tensor_list
from ipconv.models.LW_DETR.util.get_param_dicts import get_param_dict

# WEIHT_PATH = './ipconv/models/LW_DETR/LWDETR_small_60e_coco.pth'
WEIHT_PATH = './ipconv/models/LW_DETR/LWDETR_xlarge_60e_coco.pth'
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
INPUT_PATH = '00000.jpg'
OUTPUT_PATH = 'output.jpg'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_image_size = torch.tensor(image.size[::-1])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
            transforms.Resize([1024 - 512, 1024 - 256]),
            transforms.Pad([128, 256]),
            normalize,
        ])
    image = transform(image)

    orig_image_size = torch.tensor([1024, 1024])

    return image, orig_image_size

def visualize_detections(image, boxes, labels, scores, conf_thresh, output_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score > conf_thresh:
            xmin, ymin, xmax, ymax = map(int, box)

            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)

            text = f"{COCO_CLASSES[label]} {score:.2f}"
            draw.text((xmin, ymin - 10), text, fill="green", font=font)

    image.save(output_path)

def main():
    # model, _, postprocessors = build_lwdetr_small()
    model, _, postprocessors = build_lwdetr_xlarge()
    model.eval()
    model = model.to(DEVICE)

    checkpoint = torch.load(WEIHT_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)

    image, orig_image_size = preprocess_image(INPUT_PATH)
    image = image.to(DEVICE)
    orig_image_size = orig_image_size.to(DEVICE)
    print(f"Image size: {orig_image_size}")

    images = nested_tensor_from_tensor_list([image])
    orig_image_sizes = torch.stack([orig_image_size])

    # forward
    with torch.no_grad():
        import time

        for _ in range(10):
            outputs = model(images)

        num_repeats = 10
        start_time = time.time()
        for _ in range(num_repeats):
            outputs = model(images)
        end_time = time.time()
        print(f"Inference time for num_repeats iterations: {(end_time - start_time) / num_repeats:.4f} seconds per iteration")

    # postprocess
    predictions = postprocessors['bbox'](outputs, orig_image_sizes)

    # visualize
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    original_image = Image.open(INPUT_PATH).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize([1024 - 512, 1024 - 256]),
        transforms.Pad([128, 256]),
    ])
    original_image = transform(original_image)

    visualize_detections(
        original_image,
        boxes,
        labels,
        scores,
        0.5,
        OUTPUT_PATH)


if __name__ == "__main__":
    main()
    print(f"Output saved to {OUTPUT_PATH}")
    print("Validation completed successfully.")
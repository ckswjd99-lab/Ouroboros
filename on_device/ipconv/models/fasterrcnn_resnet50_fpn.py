import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os
import json

from tqdm import tqdm

from typing import Dict, List, Tuple
from collections import OrderedDict

from .constants import COCO_LABELS_LIST
from .proc_image import apply_dirtiness_map, refine_images, shift_features_dict, calculate_multi_iou, visualize_detection, graph_iou, graph_recompute

class FasterRCNN_ResNet50_FPN_Contexted(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        
        self.device = device
        self.base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        ).to(self.device)

    def create_dirtiness_map(
        self,
        anchor_image: np.ndarray, 
        current_image: np.ndarray,
        block_size: int = 32,
        dirty_thres: int = 30
    ) -> torch.Tensor:
        residual = cv2.absdiff(anchor_image, current_image)
        dirtiness_map = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

        image_H, image_W = residual.shape[:2]
        
        dirtiness_map = cv2.GaussianBlur(dirtiness_map, (7, 7), 1.5)
        dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)

        dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
        dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
        dirtiness_map = (dirtiness_map > 0).astype(np.float32)
        
        dirtiness_map = torch.tensor(dirtiness_map).unsqueeze(0).unsqueeze(0).to("cuda")

        return dirtiness_map
    
    def forward(
            self, 
            image_ndarray: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        image_tensor = torchvision.transforms.functional.to_tensor(image_ndarray).to(self.device).unsqueeze(0)

        detections = self.base_model(image_tensor)

        predictions = detections[0]
        boxes = predictions["boxes"].cpu().detach().numpy()
        labels = predictions["labels"].cpu().detach().numpy()
        scores = predictions["scores"].cpu().detach().numpy()

        return boxes, labels, scores
    
    def forward_contexted(
            self, 
            current_image: np.ndarray,
            anchor_features: Dict[str, torch.Tensor] = {},
            dirtiness_map: torch.Tensor = torch.zeros(1, 1, 1, 1).to("cuda"),
        ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:

        new_cache_features = {}

        # Convert image to tensor
        image_tensor = torchvision.transforms.functional.to_tensor(current_image).to(self.device).unsqueeze(0)
        original_image_sizes: List[Tuple[int, int]] = [(current_image.shape[0], current_image.shape[1])]

        # Preprocess the input image
        images, _ = self.base_model.transform(image_tensor, None)

        # Extract features
        x = images.tensors

        # Get the outputs of all layers
        fname = "input"
        new_cache_features[fname] = x
        x, dirtiness_map = apply_dirtiness_map(fname, x, anchor_features, dirtiness_map)

        conv1_out = self.base_model.backbone.body.conv1(x)
        conv1_out = self.base_model.backbone.body.bn1(conv1_out)
        conv1_out = self.base_model.backbone.body.relu(conv1_out)
        conv1_out = self.base_model.backbone.body.maxpool(conv1_out)
        fname = "conv1"
        new_cache_features[fname] = conv1_out
        conv1_out, dirtiness_map = apply_dirtiness_map(fname, conv1_out, anchor_features, dirtiness_map)

        layer1_out = self.base_model.backbone.body.layer1(conv1_out)
        fname = "layer1"
        new_cache_features[fname] = layer1_out
        layer1_out, dirtiness_map = apply_dirtiness_map(fname, layer1_out, anchor_features, dirtiness_map)

        layer2_out = self.base_model.backbone.body.layer2(layer1_out)
        fname = "layer2"
        new_cache_features[fname] = layer2_out
        layer2_out, dirtiness_map = apply_dirtiness_map(fname, layer2_out, anchor_features, dirtiness_map)

        layer3_out = self.base_model.backbone.body.layer3(layer2_out)
        fname = "layer3"
        new_cache_features[fname] = layer3_out
        layer3_out, dirtiness_map = apply_dirtiness_map(fname, layer3_out, anchor_features, dirtiness_map)

        layer4_out = self.base_model.backbone.body.layer4(layer3_out)
        fname = "layer4"
        new_cache_features[fname] = layer4_out
        layer4_out, dirtiness_map = apply_dirtiness_map(fname, layer4_out, anchor_features, dirtiness_map)

        # Use all layer outputs as features
        raw_features = OrderedDict([
            ("0", layer1_out),
            ("1", layer2_out),
            ("2", layer3_out),
            ("3", layer4_out),
        ])

        features = self.base_model.backbone.fpn(raw_features)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])  # Handle single tensor case

        # Detect objects
        proposals, _ = self.base_model.rpn(images, features, None)  # targets are None for inference
        detections, _ = self.base_model.roi_heads(features, proposals, images.image_sizes, None)  # targets are None for inference

        detections = self.base_model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        prediction = detections[0]  # Get the prediction for the first image in the batch

        # Extract boxes, labels, and scores
        boxes = prediction["boxes"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()

        return (boxes, labels, scores), new_cache_features
    
    @torch.no_grad()
    def validate_DAVIS(self, sequence_name, gop, data_root="/data/DAVIS", output_root="./output", leave=False):
        self.base_model.eval()

        sequence_path = os.path.join(data_root, "JPEGImages/480p", sequence_name)
        frames = sorted(os.listdir(sequence_path))

        annotations_path = os.path.join(data_root, "Annotations_bbox/480p", f"{sequence_name}.json")
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        output_path = os.path.join(output_root, "contexted_inference", sequence_name)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "temp"), exist_ok=True)

        anchor_image = None
        anchor_features_dict = None

        compute_rates = []
        inference_results = {}
        IoU_gt_results = []
        IoU_full_results = []

        pbar = tqdm(range(len(frames)), leave=leave)
        for i in pbar:
            basename = os.path.splitext(frames[i])[0]
            target_image = cv2.imread(os.path.join(sequence_path, frames[i]))
            annotation = annotations.get(basename, [])  # List of bounding boxes, each box is in a format of {'x_min': 431, 'y_min': 230, 'x_max': 460, 'y_max': 260, 'label': '14'}

            boxes_gt = [[float(box['x_min']), float(box['y_min']), float(box['x_max']), float(box['y_max'])] for box in annotation]
            labels_gt = [-1 for box in annotation]
            scores_gt = [1.0 for _ in annotation]

            (boxes_full, labels_full, scores_full), _ = self.forward_contexted(target_image)

            dirtiness_map_cache = None
            force_recompute = False

            if i % gop == 0 or force_recompute:
                force_recompute = False
                
                (boxes, labels, scores), features = self.forward_contexted(target_image)

                anchor_image = target_image
                anchor_features_dict = features

                recompute_rate = 1
            else:
                aligned_image, shift_vector = refine_images(anchor_image, target_image)

                if aligned_image is None:
                    aligned_image = np.zeros_like(target_image)
                    aligned_features = {}
                    dirtiness_map = torch.ones((1, 1, target_image.shape[0] // 32, target_image.shape[1] // 32)).to("cuda")
                else:
                    aligned_features = shift_features_dict(
                        anchor_features_dict, 
                        (aligned_image.shape[1], aligned_image.shape[0]), 
                        shift_vector
                    )
                    dirtiness_map = self.create_dirtiness_map(aligned_image, target_image, block_size=32, dirty_thres=30)

                recompute_rate = torch.mean(dirtiness_map).item()

                (boxes, labels, scores), features = self.forward_contexted(target_image, aligned_features, dirtiness_map)
                inference_results[basename] = (boxes, labels, scores)

                dirtiness_map = torch.nn.functional.interpolate(dirtiness_map, size=target_image.shape[:2], mode='nearest')
                dirtiness_map = dirtiness_map.squeeze(0).squeeze(0).cpu().numpy()
                dirtiness_map = np.stack([dirtiness_map] * 3, axis=-1)
                
                anchor_image = (target_image * dirtiness_map + aligned_image * (1 - dirtiness_map)).astype(np.uint8)
                anchor_features_dict = features

                dirtiness_map_cache = dirtiness_map

            # Calculate IoU of the boxes
            iou_gt = np.mean(calculate_multi_iou(boxes_gt, labels_gt, boxes, labels))
            IoU_gt_results.append(iou_gt if iou_gt > 0 else 0)

            iou_full = np.mean(calculate_multi_iou(boxes_full, labels_full, boxes, labels))
            IoU_full_results.append(iou_full if iou_full > 0 else 0)

            compute_rates.append(recompute_rate)
            pbar.set_description(f"Processing {basename}, recomp: {recompute_rate:.3f}, IoU (gt): {iou_gt:.3f}")

            # boost green channel of dirty area of the image
            if dirtiness_map_cache is None:
                dirtiness_map_cache = np.ones_like(target_image)
            target_image = target_image.astype(np.uint16)
            target_image[..., 1] = np.clip(target_image[..., 1] + dirtiness_map_cache[..., 1] * 50, 0, 255)
            target_image = target_image.astype(np.uint8)

            image_bbox_gt = visualize_detection(target_image, boxes_gt, labels_gt, scores_gt, colors=np.array([[0, 0, 255] for _ in range(len(COCO_LABELS_LIST))]))
            image_bbox = visualize_detection(image_bbox_gt, boxes, labels, scores)
            cv2.imwrite(os.path.join(output_path, "temp", f"{basename}.jpg"), image_bbox)

        # statistics
        avg_compute_rate = np.mean(compute_rates)
        avg_iou_gt = np.mean(IoU_gt_results)
        avg_iou_full = np.mean(IoU_full_results)

        # save graph of recompute rate and IoU
        graph_iou(IoU_gt_results, IoU_full_results, sequence_name, gop, output_path)
        graph_recompute(compute_rates, sequence_name, gop, output_path)


        # Make video of the results
        video_path = os.path.join(output_root, f"contexted_inference/{sequence_name}", f"gop{gop}.mp4")
        os.system(f"ffmpeg -y -r 10 -i {output_path}/temp/%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p {video_path} > /dev/null 2>&1")
        os.system(f"rm -rf {output_path}/temp")


        return avg_compute_rate, avg_iou_gt, avg_iou_full, inference_results

    
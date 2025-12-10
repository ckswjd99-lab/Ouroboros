from .lwdetr import build_lwdetr_xlarge
from .util.misc import NestedTensor, nested_tensor_from_tensor_list

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from PIL import Image
from torchvision import transforms


class LWDETR_xLarge_Contexted(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.model, self.criterion, self.postprocessors = build_lwdetr_xlarge()
        self.model.eval()
        self.model = self.model.to(self.device)

        # COCO label list (same as in DINO for consistency)
        self.COCO_LABELS_LIST = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        np.random.seed(42)
        self.COCO_COLORS_ARRAY = np.random.randint(256, size=(91, 3)) / 255
        self.COCO_LABELS_MAP = {k: v for v, k in enumerate(self.COCO_LABELS_LIST)}

    def preprocess_image(self, image_ndarray: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(image_ndarray).convert("RGB")
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = normalize(image)
        orig_image_size = torch.tensor([image.shape[0], image.shape[1]], dtype=torch.float)
        return image, orig_image_size

    @torch.no_grad()
    def forward_contexted(
        self,
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
        """
        image_ndarray: np.ndarray, shape (H, W, C), uint8
        anchor_features: dict, optional, for context features (not used in default LWDETR)
        only_backbone: bool, if True, only run backbone (not used in default LWDETR)
        Returns: (boxes, labels, scores), {}
        """
        image, orig_image_size = self.preprocess_image(image_ndarray)
        image = image.to(self.device)
        orig_image_size = orig_image_size.to(self.device)

        samples = nested_tensor_from_tensor_list([image])
        orig_image_sizes = torch.stack([orig_image_size])

        outputs = self.model(samples)
        predictions = self.postprocessors['bbox'](outputs, orig_image_sizes)

        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        return (boxes, labels, scores), {}
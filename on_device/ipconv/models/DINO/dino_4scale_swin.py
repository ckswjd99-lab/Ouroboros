from .main import build_model_main
from .util.slconfig import SLConfig
from .util.misc import NestedTensor, nested_tensor_from_tensor_list
from .models.dino.dino import DINO

import os, torch, numpy as np
import torch.nn as nn
from torchvision import transforms

from PIL import Image

from typing import List, Dict, Tuple


def build_dino_4scale_swin(device='cuda'):
    """
    Build the DINO model with 4-scale Swin backbone.
    """
    model_config_path = os.path.join(os.path.dirname(__file__), "config", "DINO", "DINO_4scale_swin.py")
    model_checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint0029_4scale_swin.pth")

    config = SLConfig.fromfile(model_config_path)
    config.device = device
    model, criterion, postprocessors = build_model_main(config)
    
    # Load the pre-trained weights
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    
    return model, criterion, postprocessors


class DINO_4Scale_Swin_Contexted(nn.Module):
    def __init__(self, device='cuda'):
        super(DINO_4Scale_Swin_Contexted, self).__init__()
        self.device = device
        self.model, self.criterion, self.postprocessors = build_dino_4scale_swin(device=device)
        self.model.eval()
        self.model: DINO = self.model.to(self.device)

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


        constants = dict(
            imagenet_rgb256_mean=[123.675, 116.28, 103.53],
            imagenet_rgb256_std=[58.395, 57.12, 57.375],
        )

    def preprocess_image(self, image_ndarray: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(image_ndarray).convert("RGB")

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform = transforms.Compose([
                # transforms.Resize([1024 - 512, 1024 - 256]),
                # transforms.Pad((128, 256)),
                normalize,
            ])
        image = transform(image)

        # put the image in the center of a 1024x1024 square
        image_padded = torch.zeros((3, 1024, 1024), dtype=image.dtype, device=image.device)

        center_shift_x = (1024 - image.shape[1]) // 2
        center_shift_y = (1024 - image.shape[2]) // 2
        image_padded[:, center_shift_x:center_shift_x + image.shape[1], center_shift_y:center_shift_y + image.shape[2]] = image

        orig_image_size = torch.Tensor((1024, 1024), device=image.device)

        return image, orig_image_size

    def forward(self, samples: NestedTensor, targets:List=None):
        return self.model(samples, targets)
        
    @torch.no_grad()
    def forward_contexted(
        self, 
        image_ndarray: np.ndarray, 
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
        # image_ndarray: (H, W, C)
        
        new_cache_features = {}

        image, orig_image_size = self.preprocess_image(image_ndarray)
        image = image.to(self.device)
        orig_image_size = orig_image_size.to(self.device)

        images = nested_tensor_from_tensor_list([image])
        orig_image_sizes = torch.stack([orig_image_size])

        # forward
        # outputs = self.model(images)
        outputs, new_cache_features = self.model.forward_contexted(
            images,
            cache_prefix="model",
            anchor_features=anchor_features,
            new_cache_features=new_cache_features,
            dirtiness_map=dirtiness_map,
            only_backbone=only_backbone
        )

        # postprocess
        predictions = self.postprocessors['bbox'](outputs, orig_image_sizes)

        # visualize
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        return (boxes, labels, scores), new_cache_features
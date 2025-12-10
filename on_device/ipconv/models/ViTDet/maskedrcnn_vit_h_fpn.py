import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import einops

import numpy as np
import os
import json
import cv2
from tqdm import tqdm

from typing import List, Optional, Dict, Tuple, Union

from .modeling.backbone.vit import ViT, SimpleFeaturePyramid
from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, add_decomposed_rel_pos, partial_mlp_inference, expand_mask_neighbors
from .modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec

from .modeling.meta_arch import GeneralizedRCNN
from .modeling.anchor_generator import DefaultAnchorGenerator
from .modeling.backbone.fpn import LastLevelMaxPool
from .modeling.box_regression import Box2BoxTransform
from .modeling.matcher import Matcher
from .modeling.poolers import ROIPooler
from .modeling.proposal_generator import RPN, StandardRPNHead
from .modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)


from .structures import ImageList
from .layers import ShapeSpec
from .layers.wrappers import move_device_like, shapes_to_tensor

from ..proc_image import (
    calculate_multi_iou, calculate_iou, visualize_detection, refine_images,
    graph_iou, graph_recompute
)

from .maskedrcnn_vit_fpn import MaskedRCNN_ViT_FPN_Contexted

class MaskedRCNN_ViT_H_FPN_Contexted(MaskedRCNN_ViT_FPN_Contexted):
    def __init__(self, device="cuda"):
        super().__init__()
        self.idx = 0

        self.device = device

        # constants
        self.COCO_LABELS_LIST = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        np.random.seed(42)
        self.COCO_COLORS_ARRAY = np.random.randint(256, size=(91, 3)) / 255
        self.COCO_LABELS_MAP = {k: v for v, k in enumerate(self.COCO_LABELS_LIST)}


        constants = dict(
            imagenet_rgb256_mean=[123.675, 116.28, 103.53],
            imagenet_rgb256_std=[58.395, 57.12, 57.375],
        )

        self.embed_dim, depth, num_heads, dp = 1280, 32, 16, 0.5
        self.window_block_indexes = list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
        num_classes = 80

        # backbone
        self.backbone = SimpleFeaturePyramid(
            net = ViT(
                img_size=1024,
                patch_size=16,
                embed_dim=self.embed_dim,
                depth=depth,
                num_heads=num_heads,
                drop_path_rate=dp,
                window_size=14,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=self.window_block_indexes,
                residual_block_indexes=[],
                use_rel_pos=True,
                out_feature="last_feat",
            ),
            in_feature="last_feat",
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=1024,
        ).to(self.device)

        # model
        self.base_model = GeneralizedRCNN(
            backbone=self.backbone,
            proposal_generator = RPN(
                in_features=["p2", "p3", "p4", "p5", "p6"],
                head=StandardRPNHead(in_channels=256, num_anchors=3, conv_dims=[-1, -1]),
                anchor_generator=DefaultAnchorGenerator(
                    sizes=[[32], [64], [128], [256], [512]],
                    aspect_ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64],
                    offset=0.0,
                ),
                anchor_matcher=Matcher(
                    thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
                ),
                box2box_transform=Box2BoxTransform(weights=[1.0, 1.0, 1.0, 1.0]),
                batch_size_per_image=256,
                positive_fraction=0.5,
                pre_nms_topk=(2000, 1000),
                post_nms_topk=(1000, 1000),
                nms_thresh=0.7,
            ),
            roi_heads=StandardROIHeads(
                num_classes=num_classes,
                batch_size_per_image=512,
                positive_fraction=0.25,
                proposal_matcher=Matcher(
                    thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
                ),
                box_in_features=["p2", "p3", "p4", "p5"],
                box_pooler=ROIPooler(
                    output_size=7,
                    scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                    sampling_ratio=0,
                    pooler_type="ROIAlignV2",
                ),
                box_head=FastRCNNConvFCHead(
                    input_shape=ShapeSpec(channels=256, height=7, width=7),
                    conv_dims=[256, 256, 256, 256],
                    fc_dims=[1024],
                    conv_norm="LN"
                ),
                box_predictor=FastRCNNOutputLayers(
                    input_shape=ShapeSpec(channels=1024),
                    test_score_thresh=0.05,
                    box2box_transform=Box2BoxTransform(weights=(10, 10, 5, 5)),
                    num_classes=num_classes,
                ),
                mask_in_features=["p2", "p3", "p4", "p5"],
                mask_pooler=ROIPooler(
                    output_size=14,
                    scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                    sampling_ratio=0,
                    pooler_type="ROIAlignV2",
                ),
                mask_head=MaskRCNNConvUpsampleHead(
                    input_shape=ShapeSpec(channels=256, width=14, height=14),
                    num_classes=num_classes,
                    conv_dims=[256, 256, 256, 256, 256],
                    conv_norm="LN",
                ),
            ),
            pixel_mean=constants["imagenet_rgb256_mean"],
            pixel_std=constants["imagenet_rgb256_std"],
            input_format="RGB",
        ).to(self.device)
    
    def load_weight(self, weight_pkl_path='./model_final_7224f1.pkl'):
        with open(weight_pkl_path, 'rb') as f:
            weights = pickle.load(f)['model']

        for name, param in self.base_model.named_parameters():
            if name in weights:
                param.data.copy_(torch.tensor(weights[name]))
            else:
                print(f"Parameter {name} not found in weights")

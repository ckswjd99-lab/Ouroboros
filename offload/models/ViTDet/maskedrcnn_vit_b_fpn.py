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

fidx = 0

class MaskedRCNN_ViT_B_FPN_Contexted(nn.Module):
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

        embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        num_classes = 80

        # backbone
        self.backbone = SimpleFeaturePyramid(
            net = ViT(
                img_size=1024,
                patch_size=16,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                drop_path_rate=dp,
                window_size=14,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=[
                    # 2, 5, 8 11 for global attention
                    0,
                    1,
                    3,
                    4,
                    6,
                    7,
                    9,
                    10,
                ],
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
    
    def load_weight(self, weight_pkl_path='./model_final_61ccd1.pkl'):
        with open(weight_pkl_path, 'rb') as f:
            weights = pickle.load(f)['model']

        for name, param in self.base_model.named_parameters():
            if name in weights:
                param.data.copy_(torch.tensor(weights[name]))
            else:
                print(f"Parameter {name} not found in weights")

    def create_dirtiness_map(
        self,
        anchor_image: np.ndarray, 
        current_image: np.ndarray,
        block_size: int = 16,
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

        dirtiness_map = torch.from_numpy(dirtiness_map).to(self.device)
        dirtiness_map = dirtiness_map.unsqueeze(0).unsqueeze(-1)

        return dirtiness_map

    def forward(self, image_ndarray: np.ndarray):
        # image_ndarray: (H, W, C)
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        detections = self.base_model(input)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        return boxes, labels, scores

    def forward_analyzed(self, image_ndarray: np.ndarray):
        return self.forward(image_ndarray)

    @torch.no_grad()
    def forward_contexted(
            self, 
            image_ndarray: np.ndarray, 
            anchor_features: Dict[str, torch.Tensor] = {},
            dirtiness_map: torch.Tensor = torch.ones(1, 64, 64, 1, device="cuda"),
            only_backbone: bool = False,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
        # image_ndarray: (H, W, C)

        new_cache_feature = {}
        
        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints=self.base_model.backbone.padding_constraints,
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # > ViT
        x = net.patch_embed(images.tensor)
        if net.pos_embed is not None:
            x = x + get_abs_pos(
                net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        
        # x: Tensor(1, 64, 64, 768)
        # dirtiness_map: Tensor(1, 64, 64, 1)

        dmap_window = None
        for bidx, block in enumerate(net.blocks):
            # > EncoderBlock
            shortcut = x
            x = block.norm1(x)

            # Window partition
            dmap_block = dirtiness_map.clone() if bidx not in [] else expand_mask_neighbors(dirtiness_map)
            dmap_window = None
            if block.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, block.window_size)
                # pad_hw = (70, 70)
                # pad the dirtiness map and fill with 0
                if dmap_window is None:
                    dmap_window, _ = window_partition(dmap_block, block.window_size)

            # Attention
            x_attn = x
            B_attn, H_attn, W_attn, _ = x_attn.shape

            dmap_now = dmap_window if dmap_window is not None else dmap_block
            dmap_now_flat = dmap_now.reshape(-1)

            # partial QKV generation
            x_attn_flat = x_attn.reshape(-1, 768)
            x_attn_selected = x_attn_flat[dmap_now_flat == 1, :]
            qkv_selected = block.attn.qkv(x_attn_selected)

            qkv_flat = torch.zeros(B_attn * H_attn * W_attn, 3 * 768, device=self.device, dtype=x_attn.dtype)
            qkv_flat[dmap_now_flat == 1, :] = qkv_selected

            qkv = qkv_flat.reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # qkv with shape (3, B_attn, nHead, H_attn * W_attn, C)

            fname = f"block{bidx}_qkv"
            if fname in anchor_features:
                dmap_channeled = dmap_now.reshape(B_attn, H_attn * W_attn)
                dmap_broadcastable = dmap_channeled.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                qkv = qkv * dmap_broadcastable + anchor_features[fname] * (1 - dmap_broadcastable)
            new_cache_feature[fname] = qkv.clone()

            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)  # q, k, v with shape (B_attn * nHead, H_attn * W_attn, C)

            # partial attention
            if bidx not in [2, 5, 8, 11]:   # window attention
                attn = (q * block.attn.scale) @ k.transpose(-2, -1)

                if block.attn.use_rel_pos:
                    attn = add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

                # projection
                attn = attn.softmax(dim=-1)
                x_attn = (attn @ v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)

                x_attn_flat = x_attn.reshape(-1, x_attn.shape[-1])
                x_attn_selected = x_attn_flat[dmap_now_flat == 1, :]
                x_attn_selected = block.attn.proj(x_attn_selected)
                x_attn = torch.zeros(B_attn * H_attn * W_attn, x_attn_selected.shape[-1], device=self.device, dtype=x_attn.dtype)
                x_attn[dmap_now_flat == 1, :] = x_attn_selected.view(-1, x_attn_selected.shape[-1])
                x_attn = x_attn.view(B_attn, H_attn, W_attn, -1)

            else:   # global attention
                q_selected = q[:, dmap_now_flat == 1, :]
                num_selected = q_selected.shape[1]

                attn_selected = (q_selected * block.attn.scale) @ k.transpose(-2, -1)
                attn = torch.zeros(B_attn * block.attn.num_heads, H_attn * W_attn, H_attn * W_attn, device=self.device, dtype=x_attn.dtype)
                attn[:, dmap_now_flat == 1, :] = attn_selected

                if block.attn.use_rel_pos:
                    attn = add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn), dmap_now)

                # projection
                attn_selected = attn[:, dmap_now_flat == 1, :].softmax(dim=-1)
                x_attn_selected = (attn_selected @ v).view(B_attn, block.attn.num_heads, num_selected, -1).permute(0, 2, 1, 3).reshape(B_attn, num_selected, -1)
                x_attn_selected = block.attn.proj(x_attn_selected)

                x_attn = torch.zeros(B_attn, H_attn * W_attn, x_attn_selected.shape[-1], device=self.device, dtype=x_attn.dtype)
                x_attn[:, dmap_now_flat == 1, :] = x_attn_selected
                x_attn = x_attn.view(B_attn, H_attn, W_attn, -1)

            x = x_attn
            
            # Reverse window partition
            if block.window_size > 0:
                x = window_unpartition(x, block.window_size, pad_hw, (H, W))

            # Residual
            x = shortcut + block.drop_path(x)

            shortcut2 = x
            x_norm2 = block.norm2(x)

            x_mlp_out = partial_mlp_inference(
                x_norm2,           # (B, H, W, C)
                dmap_block,        # (B, H, W, 1)
                block.mlp, 
                block.drop_path
            )
            x = shortcut2 + x_mlp_out


            if block.use_residual_block:    # nothing
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
            fname = f"block{bidx}_out"
            if fname in anchor_features:
                # x: (1, 64, 64, 768), anchor_features[fname]: (1, 64, 64, 768)
                dmap_channeled = dmap_block.expand(-1, -1, -1, x.shape[-1])    # (1, 64, 64, 768)
                x = x * dmap_channeled + anchor_features[fname] * (1 - dmap_channeled)
            new_cache_feature[fname] = x.clone()

        if only_backbone:
            return (np.array([[]]), np.array([]), np.array([])), new_cache_feature

        # > FPN
        bottom_up_features = {net._out_features[0]: x.permute(0, 3, 1, 2)}

        features = bottom_up_features[backbone.in_feature]  # (1, 768, 64, 64)
        results = []
        for stage in backbone.stages:
            results.append(stage(features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        assert len(backbone._out_features) == len(results)
        features = {f: res for f, res in zip(backbone._out_features, results)}

        # inference: RPN

        # > proposal_generator
        pgen = self.base_model.proposal_generator

        pgen_features = [features[f] for f in pgen.in_features]
        pgen_anchors = pgen.anchor_generator(pgen_features)

        # pgen_logits, pgen_deltas = pgen.rpn_head(pgen_features) # 15 ms
        pgen_logits = []
        pgen_deltas = []
        for feature in pgen_features:
            t = pgen.rpn_head.conv(feature)

            logits = pgen.rpn_head.objectness_logits(t)
            deltas = pgen.rpn_head.anchor_deltas(t)

            pgen_logits.append(logits)
            pgen_deltas.append(deltas)

        pgen_logits = [logits.permute(0, 2, 3, 1).flatten(1) for logits in pgen_logits]
        pgen_deltas = [
            deltas.view(
                deltas.shape[0],
                -1,
                pgen.anchor_generator.box_dim,
                deltas.shape[-2],
                deltas.shape[-1]
            )
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for deltas in pgen_deltas
        ]

        proposals = pgen.predict_proposals(
            pgen_anchors, pgen_logits, pgen_deltas, images.image_sizes
        )
        proposals = [proposals[0].to(self.device)]

        # > roi_heads
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # postprocess
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        # boxes, labels, scores = [], [], []

        return (boxes, labels, scores), new_cache_feature
    
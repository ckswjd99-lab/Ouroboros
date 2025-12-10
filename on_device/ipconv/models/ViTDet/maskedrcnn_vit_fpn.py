import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import numpy as np
import os
import json
import cv2
from tqdm import tqdm

from typing import List, Optional, Dict, Tuple, Union

from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, add_decomposed_rel_pos, partial_mlp_inference, AddDecomposedRelPos

from .eventful_transformer.base import ExtendedModule
from .eventful_transformer.counting import CountedAdd, CountedMatmul


from .structures import ImageList
from .layers import ShapeSpec
from .layers.wrappers import move_device_like, shapes_to_tensor

from ..proc_image import (
    calculate_multi_iou, calculate_iou, visualize_detection, refine_images,
    graph_iou, graph_recompute
)

class MaskedRCNN_ViT_FPN_Contexted(ExtendedModule):
    def __init__(self, device="cuda:0", dataset_name="coco"):
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
        self.VID_LABELS_LIST = [
            "airplane", "antelope", "bear", "bicycle", "bird",
            "bus", "car", "cattle", "dog", "domestic_cat",
            "elephant", "fox", "giant_panda", "hamster", "horse",
            "lion", "lizard", "monkey", "motorcycle", "rabbit",
            "red_panda", "sheep", "snake", "squirrel", "tiger",
            "train", "turtle", "watercraft", "whale", "zebra",
        ]

        np.random.seed(42)
        self.COCO_COLORS_ARRAY = np.random.randint(256, size=(91, 3)) / 255
        self.COCO_LABELS_MAP = {k: v for v, k in enumerate(self.COCO_LABELS_LIST)}


        constants = dict(
            imagenet_rgb256_mean=[123.675, 116.28, 103.53],
            imagenet_rgb256_std=[58.395, 57.12, 57.375],
        )

        self.embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        num_classes = 80

        # backbone
        self.backbone = nn.Identity().to(self.device)

        # model
        self.base_model = nn.Identity().to(self.device)

        # counting module
        self.add = CountedAdd()
        self.matmul = CountedMatmul()
        self.add_decomposed_rel_pos = AddDecomposedRelPos()

    @torch.no_grad()
    def forward(self, image_ndarray: np.ndarray, *args, **kwargs):
        only_backbone = kwargs.get("only_backbone", False)

        # image_ndarray: (H, W, C)
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        # detections = self.base_model(input)
        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        features = self.base_model.backbone(images.tensor)
        
        if only_backbone:
            return ([], [], []), {}
        
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        detections = self._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        return boxes, labels, scores

    @torch.no_grad()
    def forward_contexted(
            self, 
            image_ndarray: np.ndarray, 
            anchor_features: Dict[str, torch.Tensor] = {},
            dirtiness_map: torch.Tensor = torch.ones(1, 64, 64, 1, device="cuda:0"),
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
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # > ViT
        x = net.patch_embed(images.tensor)
        ape = get_abs_pos(
            net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )
        if net.pos_embed is not None:
            x = self.add(x, ape)
        
        # x: Tensor(1, 64, 64, 768)
        # dirtiness_map: Tensor(1, 64, 64, 1)

        dmap_block = dirtiness_map.clone()
        dmap_window, _ = window_partition(dmap_block, net.blocks[0].window_size)

        dindice_block = torch.nonzero(dmap_block.view(-1) == 1, as_tuple=False).squeeze(-1)
        dindice_window = torch.nonzero(dmap_window.view(-1) == 1, as_tuple=False).squeeze(-1)

        for bidx, block in enumerate(net.blocks):
            # > EncoderBlock
            shortcut = x

            x_std = torch.std(x, dim=-1, keepdim=True)
            x = block.norm1(x)

            # Window partition
            if block.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, block.window_size)
                # pad_hw = (70, 70)
                # pad the dirtiness map and fill with 0

            # Attention
            x_attn = x
            B_attn, H_attn, W_attn, _ = x_attn.shape

            dmap_now = dmap_window if block.window_size > 0 else dmap_block
            selected_indices = dindice_window if block.window_size > 0 else dindice_block

            # partial QKV generation
            x_attn_flat = x_attn.reshape(-1, self.embed_dim)
            x_attn_selected = F.embedding(selected_indices, x_attn_flat)
            qkv_selected = block.attn.qkv(x_attn_selected)

            qkv_flat = torch.zeros(B_attn * H_attn * W_attn, 3 * self.embed_dim, device=self.device, dtype=x_attn.dtype)
            qkv_flat[selected_indices, :] = qkv_selected

            qkv = qkv_flat.reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # qkv with shape (3, B_attn, nHead, H_attn * W_attn, C)

            fname = f"block{bidx}_qkv"
            if fname in anchor_features:
                dmap_channeled = dmap_now.reshape(B_attn, H_attn * W_attn)
                dmap_broadcastable = dmap_channeled.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                kv_cached = anchor_features[fname]
                qkv_cached = torch.zeros_like(qkv)
                # qkv_cached[1:] = self.add(qkv[1:] * dmap_broadcastable, kv_cached * (1 - dmap_broadcastable))
                qkv_cached = self.add(qkv * dmap_broadcastable, kv_cached * (1 - dmap_broadcastable))
                qkv = qkv_cached
            # new_cache_feature[fname] = qkv.clone()[1:]
            new_cache_feature[fname] = qkv.clone()

            fname = f"block{bidx}_qkvpe"
            if fname in anchor_features:
                new_cache_feature[fname] = anchor_features[fname]
            else:
                if block.window_size > 0:
                    ape_block, _ = window_partition(ape, block.window_size)
                else:
                    ape_block = ape
                ape_block = ape_block * block.norm1.weight
                # disable for latency measurement: can be done offline
                # ape_block = block.attn.qkv(ape_block).reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # ape_block with shape (3, B_attn, nHead, H_attn * W_attn, C)
                
                # new_cache_feature[fname] = ape_block.clone()[1:]  # for strict cache size management
                new_cache_feature[fname] = ape_block.clone() # for easy inference

            fname = f"block{bidx}_std"
            x_std, _ = window_partition(x_std, block.window_size) if block.window_size > 0 else (x_std, None)
            new_cache_feature[fname] = x_std.clone()

            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)  # q, k, v with shape (B_attn * nHead, H_attn * W_attn, C)

            # partial attention
            if bidx in self.window_block_indexes:   # window attention
                attn = self.matmul((q * block.attn.scale), k.transpose(-2, -1))

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

                # projection
                attn = attn.softmax(dim=-1)
                x_attn = self.matmul(attn, v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)

                x_attn_flat = x_attn.reshape(-1, x_attn.shape[-1])
                x_attn_selected = F.embedding(selected_indices, x_attn_flat)
                x_attn_selected = block.attn.proj(x_attn_selected)
                x_attn = torch.zeros(B_attn * H_attn * W_attn, x_attn_selected.shape[-1], device=self.device, dtype=x_attn.dtype)
                x_attn[selected_indices, :] = x_attn_selected.view(-1, x_attn_selected.shape[-1])
                x_attn = x_attn.view(B_attn, H_attn, W_attn, -1)

            else:   # global attention
                q_selected = q[:, selected_indices, :]
                num_selected = q_selected.shape[1]

                attn_selected = self.matmul((q_selected * block.attn.scale), k.transpose(-2, -1))
                attn = torch.zeros(B_attn * block.attn.num_heads, H_attn * W_attn, H_attn * W_attn, device=self.device, dtype=x_attn.dtype)
                attn[:, selected_indices, :] = attn_selected

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn), dmap_now)

                # projection
                attn_selected = attn[:, selected_indices, :].softmax(dim=-1)
                x_attn_selected = self.matmul(attn_selected, v).view(B_attn, block.attn.num_heads, num_selected, -1).permute(0, 2, 1, 3).reshape(B_attn, num_selected, -1)
                x_attn_selected = block.attn.proj(x_attn_selected)

                x_attn = torch.zeros(B_attn, H_attn * W_attn, x_attn_selected.shape[-1], device=self.device, dtype=x_attn.dtype)
                x_attn[:, selected_indices, :] = x_attn_selected
                x_attn = x_attn.view(B_attn, H_attn, W_attn, -1)

            x = x_attn
            
            # Reverse window partition
            if block.window_size > 0:
                x = window_unpartition(x, block.window_size, pad_hw, (H, W))

            # Residual
            x = self.add(shortcut, block.drop_path(x))

            shortcut2 = x
            x_norm2 = block.norm2(x)

            x_mlp_out = partial_mlp_inference(
                x_norm2,           # (B, H, W, C)
                dmap_block,        # (B, H, W, 1)
                block.mlp, 
                block.drop_path
            )
            x = self.add(shortcut2, x_mlp_out)


            if block.use_residual_block:    # nothing
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
        fname = f"block_out"
        if fname in anchor_features:
            # x: (1, 64, 64, 768), anchor_features[fname]: (1, 64, 64, 768)
            dmap_channeled = dmap_block.expand(-1, -1, -1, x.shape[-1])    # (1, 64, 64, 768)
            x = self.add(x * dmap_channeled, anchor_features[fname] * (1 - dmap_channeled))
        new_cache_feature[fname] = x.clone()

        if only_backbone:
            return ([], [], []), new_cache_feature

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
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # postprocess
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks
    
    @torch.no_grad()
    def forward_eventful(
            self, 
            image_ndarray: np.ndarray, 
            anchor_features: Dict[str, torch.Tensor] = {},
            dirtiness_map: torch.Tensor = torch.ones(1, 64, 64, 1, device="cuda:0"),
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
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # > ViT
        x = net.patch_embed(images.tensor)
        ape = get_abs_pos(
            net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )
        if net.pos_embed is not None:
            x = self.add(x, ape)
        
        # x: Tensor(1, 64, 64, 768)
        # dirtiness_map: Tensor(1, 64, 64, 1)

        dmap_block = dirtiness_map
        dmap_window, _ = window_partition(dmap_block, net.blocks[0].window_size)

        dindice_block = torch.nonzero(dmap_block.view(-1) == 1, as_tuple=False).squeeze(-1)
        dindice_window = torch.nonzero(dmap_window.view(-1) == 1, as_tuple=False).squeeze(-1)

        for bidx, block in enumerate(net.blocks):
            # > EncoderBlock
            shortcut = x

            x = block.norm1(x)

            # Window partition
            if block.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, block.window_size)
                # pad_hw = (70, 70)
                # pad the dirtiness map and fill with 0

            # Attention
            x_attn = x
            B_attn, H_attn, W_attn, _ = x_attn.shape

            dmap_now = dmap_window if block.window_size > 0 else dmap_block
            selected_indices = dindice_window if block.window_size > 0 else dindice_block

            # partial QKV generation
            x_attn_flat = x_attn.reshape(-1, self.embed_dim)
            x_attn_selected = F.embedding(selected_indices, x_attn_flat)
            qkv_selected = block.attn.qkv(x_attn_selected)

            qkv_flat = torch.zeros(B_attn * H_attn * W_attn, 3 * self.embed_dim, device=self.device, dtype=x_attn.dtype)
            qkv_flat[selected_indices, :] = qkv_selected

            qkv = qkv_flat.reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # qkv with shape (3, B_attn, nHead, H_attn * W_attn, C)

            fname = f"block{bidx}_qkv"
            if fname in anchor_features:
                dmap_channeled = dmap_now.reshape(B_attn, H_attn * W_attn)
                dmap_broadcastable = dmap_channeled.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                qkv = self.add(qkv * dmap_broadcastable, anchor_features[fname] * (1 - dmap_broadcastable))
            new_cache_feature[fname] = qkv

            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)  # q, k, v with shape (B_attn * nHead, H_attn * W_attn, C)

            # partial attention
            if bidx in self.window_block_indexes:   # window attention
                attn = self.matmul((q * block.attn.scale), k.transpose(-2, -1))

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

                # projection
                attn = attn.softmax(dim=-1)
                x_attn = self.matmul(attn, v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
                x_attn = block.attn.proj(x_attn)

            else:   # global attention
                fname = f"block{bidx}_qkv"
                qkv_cache = anchor_features[fname] if fname in anchor_features else torch.zeros_like(qkv)
                q_cache, k_cache, v_cache = qkv_cache.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)

                q_selected = q[:, selected_indices, :]
                k_selected = k[:, selected_indices, :]
                v_selected = v[:, selected_indices, :]
                num_selected = q_selected.shape[1]

                attn_selected_row = self.matmul((q_selected * block.attn.scale), k.transpose(-2, -1))
                attn_selected_col = self.matmul((q * block.attn.scale), k_selected.transpose(-2, -1))
                attn = torch.zeros(B_attn * block.attn.num_heads, H_attn * W_attn, H_attn * W_attn, device=self.device, dtype=x_attn.dtype)
                attn[:, selected_indices, :] = attn_selected_row
                attn[:, :, selected_indices] = attn_selected_col

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn), dmap_now)

                attn = attn.softmax(dim=-1)

                fname = f"block{bidx}_attn"
                attn_cache = anchor_features[fname] if fname in anchor_features else None
                new_cache_feature[fname] = attn

                # Attn_V update
                fname = f"block{bidx}_attn_v"
                AV_old = anchor_features[fname] if fname in anchor_features else None
                AV_diff = self.add(attn, (-1) * attn_cache) if attn_cache is not None else attn
                AV_diff_selected = AV_diff[:, :, selected_indices]

                v_cache_selected = v_cache[:, selected_indices, :]
                v_diff_selected = self.add(v_selected, (-1) * v_cache_selected)
                v_temp_selected = self.add(v_selected, (-1) * v_diff_selected)

                AnVdiff = self.matmul(attn[:, :, selected_indices], v_diff_selected)

                AV_update = self.matmul(AV_diff_selected, v_temp_selected)

                AV = self.add(self.add(AV_old, AnVdiff), AV_update) if AV_old is not None else self.add(AnVdiff, AV_update)

                new_cache_feature[fname] = AV

                # projection
                attn_selected = AV[:, selected_indices, :]
                x_attn_selected = attn_selected.view(B_attn, block.attn.num_heads, num_selected, -1).permute(0, 2, 1, 3).reshape(B_attn, num_selected, -1)
                x_attn_selected = block.attn.proj(x_attn_selected)

                x_attn = torch.zeros(B_attn, H_attn * W_attn, x_attn_selected.shape[-1], device=self.device, dtype=x_attn.dtype)
                x_attn[:, selected_indices, :] = x_attn_selected
                x_attn = x_attn.view(B_attn, H_attn, W_attn, -1)

            x = x_attn
            
            # Reverse window partition
            if block.window_size > 0:
                x = window_unpartition(x, block.window_size, pad_hw, (H, W))

            # Residual
            x = self.add(shortcut, block.drop_path(x))

            shortcut2 = x
            x_norm2 = block.norm2(x)

            x_mlp_out = partial_mlp_inference(
                x_norm2,           # (B, H, W, C)
                dmap_block,        # (B, H, W, 1)
                block.mlp, 
                block.drop_path
            )
            x = self.add(shortcut2, x_mlp_out)


            if block.use_residual_block:    # nothing
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
        fname = f"block_out"
        if fname in anchor_features:
            # x: (1, 64, 64, 768), anchor_features[fname]: (1, 64, 64, 768)
            dmap_channeled = dmap_block.expand(-1, -1, -1, x.shape[-1])    # (1, 64, 64, 768)
            x = self.add(x * dmap_channeled, anchor_features[fname] * (1 - dmap_channeled))
        new_cache_feature[fname] = x

        if only_backbone:
            return ([], [], []), new_cache_feature

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
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # postprocess
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks

    @torch.no_grad()
    def forward_maskvd(
            self, 
            image_ndarray: np.ndarray, 
            anchor_features: Dict[str, torch.Tensor] = {},
            dirtiness_map: torch.Tensor = torch.ones(1, 64, 64, 1, device="cuda:0"),
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
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # > ViT
        x = net.patch_embed(images.tensor)
        ape = get_abs_pos(
            net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )
        if net.pos_embed is not None:
            x = self.add(x, ape)
        
        # x: Tensor(1, 64, 64, 768)
        # dirtiness_map: Tensor(1, 64, 64, 1)

        dmap_block = dirtiness_map
        dmap_window, _ = window_partition(dmap_block, net.blocks[0].window_size)

        dindice_block = torch.nonzero(dmap_block.view(-1) == 1, as_tuple=False).squeeze(-1)
        dindice_window = torch.nonzero(dmap_window.view(-1) == 1, as_tuple=False).squeeze(-1)

        for bidx, block in enumerate(net.blocks):
            # > EncoderBlock
            shortcut = x

            x_std = torch.std(x, dim=-1, keepdim=True)
            x = block.norm1(x)

            # Window partition
            if block.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, block.window_size)
                # pad_hw = (70, 70)
                # pad the dirtiness map and fill with 0

            # Attention
            x_attn = x
            B_attn, H_attn, W_attn, _ = x_attn.shape

            dmap_now = dmap_window if block.window_size > 0 else dmap_block
            selected_indices = dindice_window if block.window_size > 0 else dindice_block

            # partial QKV generation
            x_attn_flat = x_attn.reshape(-1, self.embed_dim)
            x_attn_selected = F.embedding(selected_indices, x_attn_flat)

            fname = f"block{bidx}_x_attn_flat"
            if fname in anchor_features:
                x_attn_flat_cached = anchor_features[fname]
                x_attn_flat_cached[selected_indices] = x_attn_selected
                x_attn_selected = x_attn_flat_cached
            else:
                x_attn_selected = x_attn_flat
            new_cache_feature[fname] = x_attn_selected
            
            qkv = block.attn.qkv(x_attn_selected)
            qkv = qkv.reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # qkv with shape (3, B_attn, nHead, H_attn * W_attn, C)

            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)  # q, k, v with shape (B_attn * nHead, H_attn * W_attn, C)

            # partial attention
            if bidx in self.window_block_indexes:   # window attention
                attn = self.matmul((q * block.attn.scale), k.transpose(-2, -1))

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

                # projection
                attn = attn.softmax(dim=-1)
                x_attn = self.matmul(attn, v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)

                x_attn_flat = x_attn.reshape(-1, x_attn.shape[-1])
                x_attn_selected = x_attn_flat
                x_attn_selected = block.attn.proj(x_attn_selected)
                x_attn = x_attn_selected.view(B_attn, H_attn, W_attn, -1)

            else:   # global attention
                q_selected = q[:, selected_indices, :]
                k_selected = k[:, selected_indices, :]
                num_selected = q_selected.shape[1]

                attn_selected = self.matmul((q_selected * block.attn.scale), k_selected.transpose(-2, -1))
                attn = torch.zeros(B_attn * block.attn.num_heads, H_attn * W_attn, H_attn * W_attn, device=self.device, dtype=x_attn.dtype)
                attn[:, selected_indices[:, None], selected_indices[None, :]] = attn_selected


                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn), dmap_now)

                # projection
                attn_selected = attn[:, selected_indices, :].softmax(dim=-1)
                x_attn_selected = self.matmul(attn_selected, v).view(B_attn, block.attn.num_heads, num_selected, -1).permute(0, 2, 1, 3).reshape(B_attn, num_selected, -1)
                x_attn_selected = block.attn.proj(x_attn_selected)

                x_attn = torch.zeros(B_attn, H_attn * W_attn, x_attn_selected.shape[-1], device=self.device, dtype=x_attn.dtype)
                x_attn[:, selected_indices, :] = x_attn_selected
                x_attn = x_attn.view(B_attn, H_attn, W_attn, -1)

            x = x_attn
            
            # Reverse window partition
            if block.window_size > 0:
                x = window_unpartition(x, block.window_size, pad_hw, (H, W))

            # Residual
            x = self.add(shortcut, block.drop_path(x))

            shortcut2 = x
            x_norm2 = block.norm2(x)

            x_mlp_out = partial_mlp_inference(
                x_norm2,           # (B, H, W, C)
                dmap_block,        # (B, H, W, 1)
                block.mlp, 
                block.drop_path
            )
            x = self.add(shortcut2, x_mlp_out)


            if block.use_residual_block:    # nothing
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
        fname = f"block_out"
        if fname in anchor_features:
            # x: (1, 64, 64, 768), anchor_features[fname]: (1, 64, 64, 768)
            dmap_channeled = dmap_block.expand(-1, -1, -1, x.shape[-1])    # (1, 64, 64, 768)
            x = self.add(x * dmap_channeled, anchor_features[fname] * (1 - dmap_channeled))
        new_cache_feature[fname] = x.clone()

        if only_backbone:
            return ([], [], []), new_cache_feature

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
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # postprocess
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks

    @torch.no_grad()
    def forward_stgt(
            self, 
            image_ndarray: np.ndarray, 
            anchor_features: Dict[str, torch.Tensor] = {},
            dirtiness_map: torch.Tensor = torch.ones(1, 64, 64, 1, device="cuda:0"),
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
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        # inference: backbone
        backbone = self.base_model.backbone
        net = backbone.net

        # > ViT
        x = net.patch_embed(images.tensor)
        ape = get_abs_pos(
            net.pos_embed, net.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        )
        if net.pos_embed is not None:
            x = self.add(x, ape)
        
        # x: Tensor(1, 64, 64, 768)
        # dirtiness_map: Tensor(1, 64, 64, 1)

        dmap_block = dirtiness_map.clone()
        dmap_window, _ = window_partition(dmap_block, net.blocks[0].window_size)

        dindice_block = torch.nonzero(dmap_block.view(-1) == 1, as_tuple=False).squeeze(-1)
        dindice_window = torch.nonzero(dmap_window.view(-1) == 1, as_tuple=False).squeeze(-1)

        for bidx, block in enumerate(net.blocks):
            # > EncoderBlock
            shortcut = x

            x_std = torch.std(x, dim=-1, keepdim=True)
            x = block.norm1(x)

            # Window partition
            if block.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, block.window_size)
                # pad_hw = (70, 70)
                # pad the dirtiness map and fill with 0

            # Attention
            x_attn = x
            B_attn, H_attn, W_attn, _ = x_attn.shape

            dmap_now = dmap_window if block.window_size > 0 else dmap_block
            selected_indices = dindice_window if block.window_size > 0 else dindice_block

            # partial QKV generation
            x_attn_flat = x_attn.reshape(-1, self.embed_dim)
            x_attn_selected = F.embedding(selected_indices, x_attn_flat)
            qkv_selected = block.attn.qkv(x_attn_selected)

            qkv_flat = torch.zeros(B_attn * H_attn * W_attn, 3 * self.embed_dim, device=self.device, dtype=x_attn.dtype)
            qkv_flat[selected_indices, :] = qkv_selected

            qkv = qkv_flat.reshape(B_attn, H_attn * W_attn, 3, block.attn.num_heads, -1).permute(2, 0, 3, 1, 4)   # qkv with shape (3, B_attn, nHead, H_attn * W_attn, C)

            fname = f"block{bidx}_qkv"
            if fname in anchor_features:
                dmap_channeled = dmap_now.reshape(B_attn, H_attn * W_attn)
                dmap_broadcastable = dmap_channeled.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                qkv_cached = anchor_features[fname]
                qkv_cached = self.add(qkv * dmap_broadcastable, qkv_cached * (1 - dmap_broadcastable))
                qkv = qkv_cached
            new_cache_feature[fname] = qkv.clone()

            q, k, v = qkv.reshape(3, B_attn * block.attn.num_heads, H_attn * W_attn, -1).unbind(0)  # q, k, v with shape (B_attn * nHead, H_attn * W_attn, C)

            # partial attention
            if bidx in self.window_block_indexes:   # window attention
                attn = self.matmul((q * block.attn.scale), k.transpose(-2, -1))

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

                # projection
                attn = attn.softmax(dim=-1)
                x_attn = self.matmul(attn, v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
                x_attn = block.attn.proj(x_attn)

            else:   # global attention
                attn = self.matmul((q * block.attn.scale), k.transpose(-2, -1))

                if block.attn.use_rel_pos:
                    attn = self.add_decomposed_rel_pos(attn, q, block.attn.rel_pos_h, block.attn.rel_pos_w, (H_attn, W_attn), (H_attn, W_attn))

                # projection
                attn = attn.softmax(dim=-1)
                x_attn = self.matmul(attn, v).view(B_attn, block.attn.num_heads, H_attn, W_attn, -1).permute(0, 2, 3, 1, 4).reshape(B_attn, H_attn, W_attn, -1)
                x_attn = block.attn.proj(x_attn)

            fname = f"block{bidx}_attn_proj"
            if fname in anchor_features:
                dmap_hw = dmap_channeled.reshape(B_attn, H_attn, W_attn)
                dmap_broadcastable = dmap_hw.unsqueeze(-1)
                x_attn = self.add(x_attn * dmap_broadcastable, anchor_features[fname] * (1 - dmap_broadcastable))
            new_cache_feature[fname] = x_attn.clone()

            x = x_attn
            
            # Reverse window partition
            if block.window_size > 0:
                x = window_unpartition(x, block.window_size, pad_hw, (H, W))

            # Residual
            x = self.add(shortcut, block.drop_path(x))

            shortcut2 = x
            x_norm2 = block.norm2(x)
            x_mlp_out = block.drop_path(block.mlp(x_norm2))

            x = self.add(shortcut2, x_mlp_out)

            if block.use_residual_block:    # nothing
                x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
            fname = f"block{bidx}_out"
            if fname in anchor_features:
                # x: (1, 64, 64, 768), anchor_features[fname]: (1, 64, 64, 768)
                dmap_channeled = dmap_block.expand(-1, -1, -1, x.shape[-1])    # (1, 64, 64, 768)
                x = self.add(x * dmap_channeled, anchor_features[fname] * (1 - dmap_channeled))
            new_cache_feature[fname] = x.clone()

        if only_backbone:
            return ([], [], []), new_cache_feature

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
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)

        # postprocess
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()
        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks
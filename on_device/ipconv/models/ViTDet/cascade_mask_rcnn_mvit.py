import torch
import torch.nn as nn
import numpy as np
from functools import partial

from .modeling.backbone.utils import window_partition, window_unpartition, add_decomposed_rel_pos, window_reverse, AddDecomposedRelPos, attention_pool
from .modeling.meta_arch import GeneralizedRCNN
from .modeling.backbone.mvit import MViT

from .layers import ShapeSpec
from .modeling.backbone import FPN
from .modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec
from .modeling.meta_arch import GeneralizedRCNN
from .modeling.proposal_generator import RPN, StandardRPNHead
from .modeling.anchor_generator import DefaultAnchorGenerator
from .modeling.matcher import Matcher
from .modeling.poolers import ROIPooler
from .modeling.box_regression import Box2BoxTransform
from .modeling.roi_heads import (
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
    CascadeROIHeads
)
from .modeling.backbone.utils import get_abs_pos, window_partition, window_unpartition, add_decomposed_rel_pos, partial_mlp_inference, AddDecomposedRelPos

from .eventful_transformer.base import ExtendedModule
from .eventful_transformer.counting import CountedAdd, CountedMatmul

from .structures import ImageList

import pickle
import torch.nn.functional as F

from typing import Dict


class CascadeMaskRCNN_MViT_Contexted(ExtendedModule):
    def __init__(self, num_classes=80, device="cuda"):
        super(CascadeMaskRCNN_MViT_Contexted, self).__init__()
        
        self.base_model = nn.Identity()
        self.num_classes = num_classes
        self.device = device

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

        self.add = CountedAdd()
        self.matmul = CountedMatmul()
        self.add_decomposed_rel_pos = AddDecomposedRelPos()

    def forward_fpn(self, bottom_up_features):
        backbone = self.base_model.backbone

        ## BACKBONE: FPN ##
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        assert len(backbone._out_features) == len(results)
        features = {f: res for f, res in zip(backbone._out_features, results)}

        return features

    def _forward_attn(self, attn_op, x, anchor_features={}, dmap=None, new_cached_features={}, prefix=""):
        if dmap is None:
            dmap = torch.ones(1, x.shape[1], x.shape[2], 1, device=x.device)

        dmap = dmap.permute(0, 3, 1, 2)  # (B, 1, H, W)
        dmap = F.interpolate(dmap, size=(x.shape[1], x.shape[2]), mode="area")
        dmap = (dmap > 0).float()
        dmap_indices = dmap.view(-1).nonzero(as_tuple=False).squeeze(-1)  # (N,)
        num_selected = dmap_indices.shape[0]

        B, H, W, _ = x.shape

        x_sel = F.embedding(dmap_indices, x.view(B * H * W, -1))  # (N, C)
        qkv_sel = attn_op.qkv(x_sel) # (N, 3 * C)
        
        fname = f"{prefix}qkv"
        if fname in anchor_features:
            qkv = anchor_features[fname]
            qkv[dmap_indices] = qkv_sel
        elif num_selected == B * H * W:
            qkv = qkv_sel
        else:
            qkv = torch.zeros((B * H * W, qkv_sel.shape[-1]), dtype=x.dtype, device=x.device)
            qkv[dmap_indices] = qkv_sel
        new_cached_features[fname] = qkv

        qkv = qkv.reshape(B, H, W, 3, attn_op.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        # qkv with shape (3, B, nHead, H, W, C)

        q, k, v = qkv.reshape(3, B * attn_op.num_heads, H, W, -1).unbind(0)
        # q, k, v with shape (B * nHead, H, W, C)

        q = attention_pool(q, attn_op.pool_q, attn_op.norm_q)
        k = attention_pool(k, attn_op.pool_k, attn_op.norm_k)
        v = attention_pool(v, attn_op.pool_v, attn_op.norm_v)

        if attn_op.pool_q.stride[0] == 2:
            dmap_pool_q = F.max_pool2d(dmap, kernel_size=attn_op.pool_q.kernel_size, stride=attn_op.pool_q.stride, padding=attn_op.pool_q.padding)
        else:
            dmap_pool_q = dmap
        dmap_pool_q = (dmap_pool_q > 0).float()
        dindice_pool_q = dmap_pool_q.view(-1).nonzero(as_tuple=False).squeeze(-1)  # (Nq,)

        ori_q = q
        if attn_op.window_size:
            q, q_hw_pad = window_partition(q, attn_op.q_win_size)
            k, kv_hw_pad = window_partition(k, attn_op.kv_win_size)
            v, _ = window_partition(v, attn_op.kv_win_size)
            q_hw = (attn_op.q_win_size, attn_op.q_win_size)
            kv_hw = (attn_op.kv_win_size, attn_op.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        fname = f"{prefix}attn"
        if fname in anchor_features:
            attn = anchor_features[fname]
        else:
            attn = (q * attn_op.scale) @ k.transpose(-2, -1)

            if attn_op.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, attn_op.rel_pos_h, attn_op.rel_pos_w, q_hw, kv_hw)

            attn = attn.softmax(dim=-1)
        new_cached_features[fname] = attn

        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if attn_op.window_size:
            x = window_unpartition(x, attn_op.q_win_size, q_hw_pad, ori_q.shape[1:3])

        if attn_op.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = x.view(B, attn_op.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B * H * W, -1)
        # print(x.shape, dindice_pool_q.shape, x_sel.shape)

        # x_sel = F.embedding(dindice_pool_q, x)
        # x_sel = attn_op.proj(x_sel)  # (N, C)
        # x[dindice_pool_q] = x_sel

        x = x.reshape(B, H, W, -1)

        x = attn_op.proj(x)
        
        return x

    def forward(self, image_ndarray: np.ndarray):
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        detections = self.base_model(input)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), {}, pred_masks

    def forward_analyzed(self, image_ndarray: np.ndarray):
        ## PREPROCESSING ##
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        ## BACKBONE: MViT ##
        backbone = self.base_model.backbone
        net = backbone.bottom_up

        x = images.tensor

        x = net.patch_embed(x)
        if net.pos_embed is not None:
            x = x + get_abs_pos(net.pos_embed, net.pretrain_use_cls_token, x.shape[1:3])

        bottom_up_features = {}
        stage = 2
        for bidx, block in enumerate(net.blocks):
            print(f"{bidx}: {x.shape}")

            x_norm = block.norm1(x)
            x_block = self._forward_attn(block.attn, x_norm)

            if hasattr(block, "proj"):
                x = block.proj(x_norm)
            if hasattr(block, "pool_skip"):
                x = attention_pool(x, block.pool_skip)

            x = x + block.drop_path(x_block)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
            
            # Feature Extraction
            if bidx in net._last_block_indexes:
                name = f"scale{stage}"
                if name in net._out_features:
                    x_out = getattr(net, f"{name}_norm")(x)
                    bottom_up_features[name] = x_out.permute(0, 3, 1, 2)
                stage += 1
        

        ## BACKBONE: FPN ##
        features = self.forward_fpn(bottom_up_features)

        ## POSTPROCESSING ##
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), {}, pred_masks
    
    def forward_contexted(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cached_features = anchor_features

        ## PREPROCESSING ##
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        ## BACKBONE: MViT ##
        backbone = self.base_model.backbone
        net = backbone.bottom_up

        x = images.tensor

        x = net.patch_embed(x)
        if net.pos_embed is not None:
            x = x + get_abs_pos(net.pos_embed, net.pretrain_use_cls_token, x.shape[1:3])

        bottom_up_features = {}
        stage = 2
        for bidx, block in enumerate(net.blocks):
            x_norm = block.norm1(x)
            x_block = self._forward_attn(block.attn, x_norm, anchor_features, dirtiness_map, new_cached_features, prefix=f"block{bidx}_")

            if hasattr(block, "proj"):
                x = block.proj(x_norm)
            if hasattr(block, "pool_skip"):
                x = attention_pool(x, block.pool_skip)

            x = x + block.drop_path(x_block)

            x_shortcut = x
            x = block.norm2(x)
            x = block.mlp(x)
            x = block.drop_path(x)
            x = x + x_shortcut
            
            # Feature Extraction
            if bidx in net._last_block_indexes:
                name = f"scale{stage}"
                if name in net._out_features:
                    x_out = getattr(net, f"{name}_norm")(x)
                    bottom_up_features[name] = x_out.permute(0, 3, 1, 2)
                stage += 1
        

        ## BACKBONE: FPN ##
        features = self.forward_fpn(bottom_up_features)

        ## POSTPROCESSING ##
        proposals, _ = self.base_model.proposal_generator(images, features, None)
        results, _ = self.base_model.roi_heads(images, features, proposals, None)
        detections = self.base_model._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        pred_masks = predictions["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cached_features, pred_masks
    
    def forward_eventful(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        return self.forward(image_ndarray)
    
    def forward_maskvd(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        return self.forward(image_ndarray)
    
    def forward_stgt(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        return self.forward(image_ndarray)
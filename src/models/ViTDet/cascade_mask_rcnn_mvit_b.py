import torch
import torch.nn as nn
import numpy as np
from functools import partial

from .cascade_mask_rcnn_mvit import CascadeMaskRCNN_MViT_Contexted

from .modeling.backbone.utils import window_partition, window_unpartition, add_decomposed_rel_pos, window_reverse, AddDecomposedRelPos
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

from .eventful_transformer.base import ExtendedModule
from .eventful_transformer.counting import CountedAdd, CountedMatmul

from .structures import ImageList

import pickle
import torch.nn.functional as F

from typing import Dict


def make_cascade_mask_rcnn_mvit_b():
    num_classes = 80

    mvit_b = MViT(
        embed_dim=96,
        depth=24,
        num_heads=1,
        last_block_indexes=(1, 4, 20, 23),
        residual_pooling=True,
        drop_path_rate=0.4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        out_features=("scale2", "scale3", "scale4", "scale5")
    )

    model = GeneralizedRCNN(
        backbone=FPN(
            bottom_up=mvit_b,
            in_features=("scale2", "scale3", "scale4", "scale5"),
            out_channels=256,
            top_block=LastLevelMaxPool(),
            square_pad=1024,
            norm="LN"
        ),
        proposal_generator=RPN(
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
        roi_heads=CascadeROIHeads(
            num_classes=80,
            batch_size_per_image=512,
            positive_fraction=0.25,
            box_in_features=["p2", "p3", "p4", "p5"],
            box_pooler=ROIPooler(
                output_size=7,
                scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
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
            box_heads=[
                FastRCNNConvFCHead(
                    input_shape=ShapeSpec(channels=256, height=7, width=7),
                    conv_dims=[256, 256, 256, 256],
                    fc_dims=[1024],
                    conv_norm="LN",
                )
                for _ in range(3)
            ],
            box_predictors=[
                FastRCNNOutputLayers(
                    input_shape=ShapeSpec(channels=1024),
                    test_score_thresh=0.05,
                    box2box_transform=Box2BoxTransform(weights=(w1, w1, w2, w2)),
                    cls_agnostic_bbox_reg=True,
                    num_classes=num_classes,
                )
                for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
            ],
            proposal_matchers=[
                Matcher(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
                for th in [0.5, 0.6, 0.7]
            ],

        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        input_format="BGR",
    )

    return model


class CascadeMaskRCNN_MViT_B_Contexted(CascadeMaskRCNN_MViT_Contexted):
    def __init__(self, device="cuda"):
        super(CascadeMaskRCNN_MViT_B_Contexted, self).__init__()
        self.base_model = make_cascade_mask_rcnn_mvit_b().to(device)

    def load_weight(self, weight_pkl_path='./model_final_8c3da3.pkl'):
        with open(weight_pkl_path, 'rb') as f:
            weights = pickle.load(f)['model']

        for name, param in self.base_model.named_parameters():
            if name in weights:
                param.data.copy_(torch.tensor(weights[name]))
            else:
                print(f"Parameter {name} not found in weights")
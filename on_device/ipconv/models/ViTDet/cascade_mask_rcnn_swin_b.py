import torch

from .modeling.backbone import (SwinTransformer, FPN)
from .modeling.backbone.fpn import LastLevelMaxPool, ShapeSpec
from .modeling.meta_arch import GeneralizedRCNN
from .modeling.proposal_generator import RPN, StandardRPNHead
from .modeling.anchor_generator import DefaultAnchorGenerator
from .modeling.matcher import Matcher
from .modeling.poolers import ROIPooler
from .modeling.box_regression import Box2BoxTransform
from .modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
    CascadeROIHeads
)

from .cascade_mask_rcnn_swin import CascadeMaskRCNN_Swin_Contexted

import pickle

def make_cascade_mask_rcnn_swin_b():
    num_classes = 80

    swin_b = SwinTransformer(
        depths=[2, 2, 18, 2],
        drop_path_rate=0.4,
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
    )

    model = GeneralizedRCNN(
        backbone=FPN(
            bottom_up=swin_b,
            in_features=("p0", "p1", "p2", "p3"),
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

class CascadeMaskRCNN_Swin_B_Contexted(CascadeMaskRCNN_Swin_Contexted):
    def __init__(self, device="cuda"):
        super(CascadeMaskRCNN_Swin_B_Contexted, self).__init__()
        self.base_model = make_cascade_mask_rcnn_swin_b().to(device)

    def load_weight(self, weight_pkl_path='./model_final_246a82.pkl'):
        with open(weight_pkl_path, 'rb') as f:
            weights = pickle.load(f)['model']

        for name, param in self.base_model.named_parameters():
            if name in weights:
                param.data.copy_(torch.tensor(weights[name]))
            else:
                print(f"Parameter {name} not found in weights")
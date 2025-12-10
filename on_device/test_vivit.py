from ipconv.models.ViTDet.vivit import FactorizedViViT
import torch

model = FactorizedViViT(
    classes=97,
    input_shape=[32, 3, 320, 320],
    normalize_mean=0.45,
    normalize_std=0.225,
    spatial_config={
        "depth" : 12,
        "position_encoding_size": [20, 20],
        "block_config": {
            "dim": 768,
            "heads": 12,
            "mlp_ratio": 4,
        },
    },
    spatial_views=3,
    temporal_config={
        "depth": 4,
        "position_encoding_size": [16],
        "block_config": {
            "dim": 768,
            "heads": 12,
            "mlp_ratio": 4,
        }
    },
    temporal_stride=2,
    temporal_views=4,
    tubelet_shape=[2, 16, 16],
)

model.load_state_dict(
    torch.load("./weights/vivit_b_epic_kitchens_final_200.pth")
)
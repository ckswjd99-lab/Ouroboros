#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageNet-VID val 평가 + GT/Pred 시각화 (Boxes, Label, Conf).
저장한 JPG → 시퀀스별 MP4 생성 후 JPG 삭제.
"""

import argparse, os, cv2, numpy as np, torch, shutil
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from datasets.vid import VID, VIDResize, CLASSES            # ← 클래스 이름
from ipconv.models import (
    MaskedRCNN_ViT_B_FPN_Contexted,
    MaskedRCNN_ViT_L_FPN_Contexted,
    MaskedRCNN_ViT_H_FPN_Contexted,
)
from ipconv.models.ViTDet.structures import Boxes, Instances
from ipconv.models.ViTDet.modeling.roi_heads import FastRCNNOutputLayers, MaskRCNNConvUpsampleHead
from ipconv.models.ViTDet.layers import ShapeSpec
from ipconv.models.ViTDet.modeling.box_regression import Box2BoxTransform
from ipconv.models.ViTDet.maskedrcnn_vit_b_fpn import reset_head


# ───────────────────── Helper ─────────────────────
# reset_head는 maskedrcnn_vit_b_fpn.py에서 import하여 사용

def get_wrapper(size, device):
    cls = dict(B=MaskedRCNN_ViT_B_FPN_Contexted,
               L=MaskedRCNN_ViT_L_FPN_Contexted,
               H=MaskedRCNN_ViT_H_FPN_Contexted)[size.upper()]
    return cls(device)


def draw_gt(vis, boxes, labels):
    for box, lab in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        name = CLASSES[lab]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def draw_pred(vis, boxes, labels, scores, thr=0.3):
    for box, lab, sc in zip(boxes, labels, scores):
        if sc < thr: continue
        x1, y1, x2, y2 = map(int, box)
        name = CLASSES[lab] if lab < len(CLASSES) else str(lab)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{name} {sc:.2f}", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def dict_csv_header(x):
    """
    Returns a CSV-header string containing the keys of a dict.

    :param x: A dict
    """
    return ",".join(k for k in sorted(x.keys()))

def dict_csv_line(x):
    """
    Returns a CSV-content string containing the values of a dict.

    :param x: A dict
    """
    return ",".join(f"{x[k]:g}" for k in sorted(x.keys()))

def save_csv_results(results, output_dir, first_run=False):
    for key, val in results.items():
        with open(output_dir / f"{key}.csv", "a") as csv_file:
            if first_run:
                print(dict_csv_header(val), file=csv_file)
            print(dict_csv_line(val), file=csv_file)


# ───────────────────── Main ───────────────────────
@torch.no_grad()
def main():
    # args
    pa = argparse.ArgumentParser()
    pa.add_argument("--size", choices=["B","L","H"], default="B")
    pa.add_argument("--ckpt", default=None,
                    help="state_dict of fine-tuned base_model")
    pa.add_argument("--thr", type=float, default=0.3)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    wrapper = get_wrapper(args.size, device)
    reset_head(wrapper.base_model, num_classes=30, device=device)
    if args.ckpt is None:
        args.ckpt = f"output/ckpt_{args.size.lower()}_5ep/vitdet_{args.size.lower()}_final.pth"
    
    pt_weight = torch.load(args.ckpt, map_location="cpu")
    pt_weight["roi_heads.mask_head.predictor.weight"] = torch.randn((31, 256, 1, 1))
    pt_weight["roi_heads.mask_head.predictor.bias"] = torch.randn((31))
    wrapper.base_model.load_state_dict(pt_weight, strict=False)
    wrapper.eval()

    # data
    long_edge, short_edge = 1024, 640 * 1024 // 1024
    vid_val = VID(Path("data/vid"), split="vid_val",
                  tar_path=Path("data/vid/data.tar"),
                  combined_transform=VIDResize(short_edge, long_edge))

    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=None, 
                              class_metrics=True)

    vis_root = Path(f"output/eval_{args.size.lower()}"); vis_root.mkdir(parents=True, exist_ok=True)

    num_iter = 3
    for seq_idx, sequence in enumerate(vid_val):
        seq_dir = vis_root / f"seq{seq_idx:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(sequence, batch_size=1)
        for fidx, (frame, ann) in enumerate(tqdm(loader, leave=False)):
            img = frame.squeeze(0).permute(1,2,0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes, labels, scores = wrapper(img)
            preds = {"boxes": torch.tensor(boxes, device=device),
                     "labels": torch.tensor(labels, device=device),
                     "scores": torch.tensor(scores, device=device)}
            targets = {"boxes": ann["boxes"].squeeze(0).to(device),
                       "labels": ann["labels"].squeeze(0).to(device)}
            metric.update([preds], [targets])

            vis = img.copy()
            draw_gt (vis, targets["boxes"].cpu(), targets["labels"].cpu())
            draw_pred(vis, boxes, labels, scores, args.thr)
            cv2.imwrite(str(seq_dir / f"f{fidx:04d}.jpg"), vis)

        # ffmpeg → mp4
        mp4_path = vis_root / f"seq{seq_idx:03d}.mp4"
        os.system(f'ffmpeg -y -framerate 30 -pattern_type glob '
                  f'-i "{seq_dir}/f*.jpg" '
                  f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                  f'-c:v libx264 -pix_fmt yuv420p {mp4_path} > /dev/null 2>&1')
        shutil.rmtree(seq_dir)  # JPG 삭제

        num_iter -= 1
        if num_iter <= 0: break

    stats = metric.compute()
    
    print(stats)
    


if __name__ == "__main__":
    main()
    

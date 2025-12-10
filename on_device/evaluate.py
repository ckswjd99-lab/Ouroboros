import argparse
import os
import sys
import time
import cv2
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import math
import imageio

import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from typing import List, Dict, Any, Tuple
from davis.davis2017.metrics import db_eval_iou, db_eval_boundary


from evaluate_funcs import (
    prepare_environment,
    estimate_affine,
    refresh_placing_matrix,
    shift_anchor_features,
    shift_anchor_features_swin,
    refresh_placing_matrix,
    create_dirtiness_map,
    create_sensitivity_map,
    expand_mask_neighbors
)
from ipconv.models.ViTDet.eventful_transformer.base import dict_string
from ipconv.models.ViTDet.modeling.backbone.utils import get_abs_pos

J_list = []
F_list = []
recomp_rate_list = []

global_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_sequence(
    model: nn.Module,
    sequence_name: str,
    sequence_data: List[Tuple[torch.Tensor, Dict[str, int]]],
    frame_rate: int,
    method: str,
    dataset_name: str,
    args,
    **kwargs: Any
):
    """
    Evaluate the model on a single sequence of images.
    """
    def safe_tensor(array, shape, dtype):
        if array.size > 0:
            return torch.from_numpy(array).reshape(shape).type(dtype)
        else:
            # shape 내 -1을 0으로 바꿔서 empty tensor를 안전하게 생성
            safe_shape = tuple(0 if s == -1 else s for s in shape)
            return torch.empty(*safe_shape, dtype=dtype)

    if method == "maskvd":
        img_max_size = int(1024 * 0.8) // 2 * 2

        maskvd_heatmap = np.load("maskvd_heatmap.npy")
        maskvd_heatmap = (maskvd_heatmap - maskvd_heatmap.min()) / (maskvd_heatmap.max() - maskvd_heatmap.min())

        maskvd_heatmap = cv2.resize(
            maskvd_heatmap,
            dsize=None,
            fx=img_max_size / max(maskvd_heatmap.shape[:2]),
            fy=img_max_size / max(maskvd_heatmap.shape[:2]),
            interpolation=cv2.INTER_LINEAR
        )

        hmap_H, hmap_W = maskvd_heatmap.shape[:2]
        heatmap = np.zeros((1024, 1024), dtype=np.float32)
        # place at center
        heatmap[(1024 - hmap_H) // 2:(1024 + hmap_H) // 2, (1024 - hmap_W) // 2:(1024 + hmap_W) // 2] = maskvd_heatmap
        # repeat to 3 channels
        heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)

    
    # pbar = tqdm(enumerate(sequence_data), leave=False)
    pbar = enumerate(sequence_data)
    img_sample = sequence_data[0][0]
    img_H, img_W = img_sample.shape[:2]
    input_img_size = (1024, 1024)
    block_size = 16
    background_color = kwargs.get("background_color", (0, 0, 0))
    
    centering_vector = np.array([(input_img_size[1] - img_W) / 2, (input_img_size[0] - img_H) / 2])
    centering_matrix = np.array([
        [1.0, 0.0, centering_vector[0]],
        [0.0, 1.0, centering_vector[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    if "Swin" in model.__class__.__name__:
        ape = None
    elif "MViT" in model.__class__.__name__:
        # print(model.base_model.backbone.bottom_up.pos_embed)
        # print(model.base_model.backbone.bottom_up.pretrain_use_cls_token)
        # ape = get_abs_pos(
        #     model.base_model.backbone.bottom_up.pos_embed,
        #     model.base_model.backbone.bottom_up.pretrain_use_cls_token,
        #     (input_img_size[0] // block_size, input_img_size[1] // block_size)
        # )
        ape = None
    else:
        ape = get_abs_pos(
            model.backbone.net.pos_embed,
            model.backbone.net.pretrain_use_cls_token,
            (input_img_size[0] // block_size, input_img_size[1] // block_size)
        )

    # variables
    frames_until_refresh = 0

    ref_frame = None
    ref_frame_aligned = None
    cached_features_dict = {}

    cum_shift_x, cum_shift_y = 0, 0
    placing_matrix = centering_matrix.copy()
    sensitivity_map = None

    for idx, (image, annotations) in pbar:
        image: np.ndarray
        annotations: Dict[str, int]


        ## REFRESH CHECK ##
        refresh = False
        
        # > frame rate
        if frames_until_refresh <= 0:
            refresh = True
        
        # > frame drift out
        shift_x, shift_y = 0, 0
        if not refresh:
            affine_matrix = estimate_affine(ref_frame, image)
            placing_matrix = placing_matrix @ np.vstack([affine_matrix, [0, 0, 1]])

            refresh_pmat, placing_matrix, (shift_x, shift_y) = refresh_placing_matrix(
                placing_matrix, img_H, img_W, input_img_size, block_size
            )

            refresh |= refresh_pmat
        cum_shift_x += shift_x
        cum_shift_y += shift_y

        # > scale check
        if not refresh:
            scaling_factor = np.sqrt(np.linalg.det(placing_matrix[:2, :2]))
            if scaling_factor < 0.8 or scaling_factor > 1.2:
                refresh = True


        ## PREPROCESS ##
        if refresh:
            placing_matrix = centering_matrix.copy()
            frames_until_refresh = frame_rate
            cached_features_dict = {}
            shift_x, shift_y = 0, 0
        
        if method != "ours":
            placing_matrix = centering_matrix.copy()
            shift_x, shift_y = 0, 0

        # > Place the image in the input
        image_placed = np.zeros((input_img_size[1], input_img_size[0], 3), dtype=np.uint8)
        image_placed = cv2.warpAffine(
            image, 
            placing_matrix[:2, :],
            dsize=input_img_size,
            dst=image_placed,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color
        )

        # > Shift cached features and reference frame
        if shift_x != 0 or shift_y != 0:
            if "Swin" in model.__class__.__name__:
                cached_features_dict = shift_anchor_features_swin(
                    cached_features_dict, shift_x * (block_size // 4), shift_y * (block_size // 4)
                )
            else:
                cached_features_dict = shift_anchor_features(
                    cached_features_dict, shift_x, shift_y, ape
                )
            if ref_frame_aligned is not None:
                ref_frame_aligned = np.roll(ref_frame_aligned, shift=(-shift_y * block_size, -shift_x * block_size), axis=(0, 1))
            if sensitivity_map is not None:
                sensitivity_map = np.roll(sensitivity_map, shift=(-shift_y * block_size, -shift_x * block_size), axis=(0, 1))
        
        # > Create dirtiness map and sensitivity map
        if not refresh:
            if method == "maskvd":
                dmap_raw = create_dirtiness_map(
                    anchor_image=heatmap,
                    current_image=np.zeros_like(heatmap, dtype=np.float32),
                    block_size=block_size,
                    dmap_type=args.dmap_type,
                    dirty_thres=args.dirty_thres,
                    dirty_topk=args.dirty_topk
                )
            else:
                dmap_raw = create_dirtiness_map(
                    anchor_image=ref_frame_aligned,
                    current_image=image_placed,
                    block_size=block_size,
                    dmap_type=args.dmap_type,
                    dirty_thres=args.dirty_thres,
                    dirty_topk=args.dirty_topk,
                    chromakey = np.array([0, 0, 0], dtype=np.uint8)  # black chromakey
                )

            if isinstance(dmap_raw, np.ndarray):
                dmap = torch.from_numpy(dmap_raw).to(global_device)
            elif isinstance(dmap_raw, torch.Tensor):
                dmap = dmap_raw.to(global_device)
            else:
                raise TypeError("Unsupported type for dirtiness map")
        else:
            dmap = torch.ones(1, 64, 64, 1, device=global_device)

        dmap_ndarray = dmap.squeeze().cpu().numpy()
        dmap_ndarray = cv2.resize(dmap_ndarray, (input_img_size[0], input_img_size[1]), interpolation=cv2.INTER_NEAREST)
        dmap_ndarray = np.repeat(dmap_ndarray[:, :, np.newaxis], 3, axis=2)

        # > Update the placed image with the dirtiness map
        # if ref_frame_aligned is not None:
        #     image_placed = image_placed * dmap_ndarray + ref_frame_aligned * (1 - dmap_ndarray)
        #     image_placed = np.clip(image_placed, 0, 255).astype(np.uint8)

        # > Expand the sensitive area
        if sensitivity_map is not None and method == "ours" and args.roi_expand > 0:
            dmap_expanded = expand_mask_neighbors(dmap).cpu().numpy().squeeze(0).squeeze(-1)
            sensi_map_downsized = cv2.resize(sensitivity_map, (input_img_size[0] // block_size, input_img_size[1] // block_size), interpolation=cv2.INTER_AREA)
            sensi_map_downsized = (sensi_map_downsized > 0.0).astype(np.float32)
            dmap_expanded = dmap_expanded * sensi_map_downsized + dmap.squeeze().cpu().numpy() * (1 - sensi_map_downsized)
            dmap_recompute = torch.from_numpy(dmap_expanded).unsqueeze(0).unsqueeze(-1).to(global_device)
        elif sensitivity_map is not None and method == "maskvd":
            sensi_map_downsized = cv2.resize(sensitivity_map, (input_img_size[0] // block_size, input_img_size[1] // block_size), interpolation=cv2.INTER_AREA)
            sensi_map_downsized = (sensi_map_downsized > 0.0).astype(np.float32)
            dmap_expanded = sensi_map_downsized + dmap.squeeze().cpu().numpy() * (1 - sensi_map_downsized)
            dmap_recompute = torch.from_numpy(dmap_expanded).unsqueeze(0).unsqueeze(-1).to(global_device)
        else:
            dmap_expanded = dmap.squeeze().cpu().numpy()
            dmap_recompute = dmap
        
        # > Update the placed image with the dirtiness map
        if ref_frame_aligned is not None:
            dmap_expanded = cv2.resize(dmap_expanded, (input_img_size[0], input_img_size[1]), interpolation=cv2.INTER_NEAREST)
            image_placed = image_placed * dmap_expanded[:, :, None] + ref_frame_aligned * (1 - dmap_expanded[:, :, None])
            image_placed = np.clip(image_placed, 0, 255).astype(np.uint8)


        ## INFERENCE ##
        if method == "ours":
            (boxes_cont, labels_cont, scores_cont), cached_features_dict, pred_masks = model.forward_contexted(image_placed, cached_features_dict, dmap_recompute)
        elif method == "evit":
            (boxes_cont, labels_cont, scores_cont), cached_features_dict, pred_masks = model.forward_eventful(image_placed, cached_features_dict, dmap_recompute)
        elif method == "maskvd":
            (boxes_cont, labels_cont, scores_cont), cached_features_dict, pred_masks = model.forward_maskvd(image_placed, cached_features_dict, dmap_recompute)
        elif method == "stgt":
            (boxes_cont, labels_cont, scores_cont), cached_features_dict, pred_masks = model.forward_stgt(image_placed, cached_features_dict, dmap_recompute)
        
        score_thres = 0.5
        # > Filter boxes by score threshold
        result_mask = scores_cont >= score_thres
        boxes_cont = boxes_cont[result_mask]
        labels_cont = labels_cont[result_mask]
        scores_cont = scores_cont[result_mask]
        pred_masks = pred_masks[result_mask]
        
        ## POSTPROCESS ##
        # > Create sensitivity map
        if method in ["ours", "maskvd"]:
            sensitivity_map = create_sensitivity_map(boxes_cont, scores_cont, input_img_size)
        
        # '''
        ## VISUALIZE ##
        # > Draw the full border
        vis_image = image_placed.copy()
        cv2.rectangle(vis_image, (0, 0), (input_img_size[0], input_img_size[1]), (0, 255, 255), 2)

        # > Boost the dirtiness map
        dmap_recompute = dmap_recompute.squeeze().cpu().numpy()
        dmap_recompute = cv2.resize(dmap_recompute, (input_img_size[0], input_img_size[1]), interpolation=cv2.INTER_NEAREST)
        vis_image[:, :, 1] = np.clip(vis_image[:, :, 1] + dmap_recompute * 30, 0, 255)


        # > Draw boxes and labels on the placed image
        for box, label, score in zip(boxes_cont, labels_cont, scores_cont):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"{label} {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # rolling shift the image with cum_shift_x and cum_shift_y
        vis_image = np.roll(vis_image, shift=(cum_shift_y * block_size, cum_shift_x * block_size), axis=(0, 1))
        
        if ref_frame_aligned is not None:
            cv2.imwrite(f"temp/{method}_{frame_rate}fps/{sequence_name}/{idx:04d}_ref.jpg", ref_frame_aligned[:, :, ::-1])
        cv2.imwrite(f"temp/{method}_{frame_rate}fps/{sequence_name}/{idx:04d}.jpg", vis_image[:, :, ::-1])

        #print(f"Processed frame {idx} of sequence {sequence_name}, boxes: {len(boxes_cont)}")
        # '''
        ref_frame = image.copy()
        ref_frame_aligned = image_placed.copy()
        frames_until_refresh -= 1

        # affine predicted masks
        def save_segmentation_mask(
            pred_masks: np.ndarray,           # (N, H, W)
            placing_matrix: np.ndarray,       # 3x3 or 2x3 affine matrix
            output_shape: Tuple[int, int],    # (H, W) of original image
            save_path: str,
            score_thresh: float = 0.5
        ):
            inverse_affine = np.linalg.inv(placing_matrix)[:2, :]

            composite_mask = np.zeros(output_shape, dtype=np.uint8)  # (H, W)

            for i, mask in enumerate(pred_masks):
                # (optional) threshold if mask is float
                mask = (mask > score_thresh).astype(np.uint8) * 255

                warped_mask = cv2.warpAffine(
                    mask,
                    inverse_affine,
                    dsize=(output_shape[1], output_shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # composite_mask[warped_mask > 127] = i + 1
                composite_mask[warped_mask > 127] = 255

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.imwrite(save_path, composite_mask)
        
        frame_name = f"{idx:05d}"
        output_mask_path = f"./pred_masks_{dataset_name}/{method}_{frame_rate:d}/{sequence_name}/{frame_name}.png"


        original_shape = (img_H, img_W)  # image.shape[:2] before placing

        save_segmentation_mask(
            pred_masks=pred_masks,                   # (N, H, W)
            placing_matrix=placing_matrix, 
            output_shape=original_shape,            # (H, W)
            save_path=output_mask_path
        )

        if annotations:
            gt = cv2.imread(annotations, 0)
            pred = cv2.imread(output_mask_path, 0)

            img_max_size = int(1024 * 0.8) // 2 * 2
            scale_factor = img_max_size / max(gt.shape[:2])
            gt = cv2.resize(
                gt,
                dsize=None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST
            )

            j = db_eval_iou(gt, pred)
            f = db_eval_boundary(gt, pred)

            J_list.append(j)
            F_list.append(f)
        
        recomp_rate_list.append(dmap_recompute.mean().item())


def evaluate(
    model, 
    dataset: Dict[str, List[Tuple[torch.Tensor, Dict[str, int]]]],
    frame_rates: List[int],
    method: str,
    args,
    **kwargs: Any
):
    """
    Evaluate the model on the dataset at specified frame rates.
    """
    model.eval()
    model.counting()
    model.clear_counts()
    n_frames = 0

    pbar = tqdm(dataset.items(), total=len(dataset))
    for sequence_name, sequence_data in pbar:
        if sequence_name == "name": continue

        dataset_name = "davis2017_trainval"  # Default dataset name, can be changed based on the dataset structure
        for frame_rate in frame_rates:
            # print(f"Evaluating sequence: {sequence_name}, frame rate: {frame_rate} fps")
            evaluate_sequence(model, sequence_name, sequence_data, frame_rate, method, args=args, dataset_name=dataset_name, **kwargs)
            model.reset()
            n_frames += len(sequence_data)
        
        # break

        pbar.set_description(f"meanJ {sum(J_list) / len(J_list):.4f}, meanF {sum(F_list) / len(F_list):.4f}")
    
    counts = model.total_counts() / n_frames
    model.clear_counts()

    return {"counts": counts}
    


@torch.no_grad()
def main(args):
    def tee_print(s, file, flush=True):
        print(s, flush=flush)
        print(s, file=file, flush=flush)

    model, dataset, settings_dict = prepare_environment(args)
    model.eval()

    counts = evaluate(model, dataset, args.frame_rates, args.method, args=args, **settings_dict)

    model_name = f"{args.model}"
    frame_rate_str = f"{args.frame_rates[0]}fps"
    dirtiness_key = f"thres{args.dirty_thres}" if args.dmap_type == "threshold" else f"topk{args.dirty_topk}"
    output_dir = Path("output") / args.dataset / model_name / f"{args.method}_{frame_rate_str}" / dirtiness_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 평균 계산
    mean_J = np.mean(J_list)
    mean_F = np.mean(F_list)
    mean_recomp_rate = np.mean(recomp_rate_list)

    # 3. 파일로 저장
    with open(output_dir / "mean_JF.txt", "w") as tee_file:
        tee_file.write(f"Model: {model_name}\n")
        tee_file.write(f"Frame Rate: {args.frame_rates[0]} fps\n")
        tee_file.write(f"Dirtiness Map Type: {args.dmap_type}\n")
        tee_file.write(f"Dirtiness Threshold: {args.dirty_thres}\n")
        tee_file.write(f"Dirtiness Top-K: {args.dirty_topk}\n")
        tee_file.write(f"Method: {args.method}\n")
        tee_file.write(f"\n")
        tee_file.write(f"Mean J: {mean_J:.4f}\n")
        tee_file.write(f"Mean F: {mean_F:.4f}\n")
        tee_file.write(f"\n")
        tee_file.write(f"Mean Recomp Rate: {mean_recomp_rate:.4f}\n")
        for key, val in counts.items():
            tee_print(key.capitalize(), tee_file)
            tee_print(dict_string(val), tee_file)

    print(f"[Saved] Mean J and F written to {output_dir / 'mean_JF.txt'}")
    print(f"mean J: {mean_J:.4f}, mean F: {mean_F:.4f}, mean recomp rate: {mean_recomp_rate:.4f}")


def parse_int_list(value):
    """Parse comma-separated integers into a list."""
    return [int(x.strip()) for x in value.split(',')]

def parse_str_list(value):
    """Parse comma-separated strings into a list."""
    return [x.strip() for x in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")
    parser.add_argument("--model", type=str, default="vitdet-b", help="Model to use for evaluation.",
        choices=["vitdet-b", "vitdet-l", "vitdet-h", "dino-swin4", "lwdetr", "swin-b", "swin-l", "mvit-b"],
    )
    parser.add_argument("--dataset", type=str, default="DAVIS2017_trainval", help="Dataset to evaluate on.",
        choices=["DAVIS2017_trainval", "DAVIS2019_challenge", "DAVIS2019_testdev"],
    )
    parser.add_argument("--frame-rates", type=parse_int_list, default=[30], 
                       help="Frame rate(s) for evaluation. Comma-separated integers (e.g., 1,6,100).")
    parser.add_argument("--sequence", type=parse_str_list, default=None, 
                       help="Specific sequence(s) to evaluate on. Comma-separated strings (e.g., bear,camel). If None, evaluates on all sequences.")
    parser.add_argument("--dmap-type", type=str, choices=["threshold", "topk"], default="threshold",
                       help="Type of dirtiness map to use. 'threshold' for thresholding, 'topk' for top-k dirtiness.")
    parser.add_argument("--dirty-thres", type=int, default=30, nargs="?",
                       help="Dirtiness threshold for the dirtiness map. Default is 30.")
    parser.add_argument("--dirty-topk", type=int, default=100, nargs="?",
                       help="Top-k dirtiness for the dirtiness map. Default is 100.")
    parser.add_argument("--method", type=str, choices=["ours", "evit", "maskvd", "stgt"], default="ours",
                       help="Method to use for evaluation. 'ours' for IPConv, 'evit' for Eventful ViT.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the evaluation on.")
    parser.add_argument("--roi-expand", type=int, default=1, help="ROI expand factor.")
    args = parser.parse_args()

    print(args)
    global_device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if global_device == "cpu":
        print("Warning: Running on CPU, this may be slow for large models or datasets.")
        exit(1)

    main(args)

    
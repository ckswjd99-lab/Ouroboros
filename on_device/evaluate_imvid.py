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

import torch
import torch.nn as nn
from ipconv.models.ViTDet.eventful_transformer.base import dict_csv_header, dict_csv_line, dict_string
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from typing import List, Dict, Any, Tuple

from evaluate_funcs import (
    prepare_environment,
    estimate_affine,
    refresh_placing_matrix,
    shift_anchor_features,
    shift_anchor_features_swin,
    refresh_placing_matrix,
    create_dirtiness_map,
    create_sensitivity_map,
    expand_mask_neighbors,
    objdet_coco_to_imvid
)
from ipconv.models.ViTDet.modeling.backbone.utils import get_abs_pos

outputs = []
labels = []
recompute_rate = []
global_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_sequence(
    model: nn.Module,
    sequence_name: str,
    sequence_data: List[Tuple[torch.Tensor, Dict[str, int]]],
    frame_rate: int,
    method: str,
    args,
    dmap_type: str = "threshold",
    dirty_thres: int = 30,
    dirty_topk: int = 100,
    sensi_expansion: int = 1,
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

        maskvd_heatmap = np.load("maskvd_heatmap_vid.npy")
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

    
    # pbar = tqdm(enumerate(sequence_data), leave=False, total=len(sequence_data), desc=f"Evaluating {sequence_name} at {frame_rate} fps")
    pbar = enumerate(sequence_data)
    img_sample = sequence_data[0][0]
    img_H, img_W = img_sample.shape[1:]
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

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

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
            #print(f"Scaling factor: {scaling_factor:.2f}")
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
            elif "MViT" in model.__class__.__name__:
                pass
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
                    dmap_type=dmap_type,
                    dirty_thres=dirty_thres,
                    dirty_topk=dirty_topk
                )
            else:
                dmap_raw = create_dirtiness_map(
                    anchor_image=ref_frame_aligned,
                    current_image=image_placed,
                    block_size=block_size,
                    dmap_type=dmap_type,
                    dirty_thres=dirty_thres,
                    dirty_topk=dirty_topk
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
        
        boxes_cont, labels_cont, scores_cont = objdet_coco_to_imvid(boxes_cont, labels_cont, scores_cont)
        
        ## POSTPROCESS ##
        # > Create sensitivity map
        if method in ["ours", "maskvd"]:
            sensitivity_map = create_sensitivity_map(boxes_cont, scores_cont, input_img_size)
        
        '''
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
            cv2.imwrite(f"temp/{sequence_name}_{idx:04d}_ref.jpg", ref_frame_aligned[:, :, ::-1])
        cv2.imwrite(f"temp/{sequence_name}_{idx:04d}.jpg", vis_image[:, :, ::-1])

        #print(f"Processed frame {idx} of sequence {sequence_name}, boxes: {len(boxes_cont)}")
        # '''
        ref_frame = image.copy()
        ref_frame_aligned = image_placed.copy()
        frames_until_refresh -= 1
        
        # affine predicted bounding box
        def inverse_affine_boxes(transformed_boxes, placing_matrix):
            inverse_affine = np.linalg.inv(placing_matrix)[:2, :]

            restored_boxes = []
            for box in transformed_boxes:
                x1, y1, x2, y2 = box

                point_lt = np.array([x1, y1], dtype=np.float32).reshape(-1, 1, 2)
                point_rt = np.array([x2, y1], dtype=np.float32).reshape(-1, 1, 2)
                point_lb = np.array([x1, y2], dtype=np.float32).reshape(-1, 1, 2)
                point_rb = np.array([x2, y2], dtype=np.float32).reshape(-1, 1, 2)

                src_pts = np.concatenate([point_lt, point_rt, point_lb, point_rb], axis=0)
                dst_pts = cv2.transform(src_pts, inverse_affine)

                x_min = int(np.mean(dst_pts[[0, 2], 0, 0]))
                y_min = int(np.mean(dst_pts[[0, 1], 0, 1]))
                x_max = int(np.mean(dst_pts[[1, 3], 0, 0]))
                y_max = int(np.mean(dst_pts[[2, 3], 0, 1]))

                restored_boxes.append([x_min, y_min, x_max, y_max])
            return restored_boxes
        
        boxes_affined = inverse_affine_boxes(boxes_cont, placing_matrix)
        boxes_affined = np.array(boxes_affined, dtype=np.float32)


        result = {
            "boxes": safe_tensor(boxes_affined, (-1, 4), torch.float32),
            "labels": safe_tensor(labels_cont, (-1,), torch.int64),
            "scores": safe_tensor(scores_cont, (-1,), torch.float32)
        }

        outputs.append(result)
        gt_boxes = annotations["boxes"].reshape(-1, 4)
        gt_labels = annotations["labels"].reshape(-1)
        labels.append({"boxes": gt_boxes, "labels": gt_labels})
        recompute_rate.append(dmap_recompute.sum().item() / dmap_recompute.numel())

    #os.system(f"ffmpeg -framerate {frame_rate} -i temp/{sequence_name}_%04d.jpg -c:v libx264 -pix_fmt yuv420p temp/{sequence_name}_{frame_rate}fps.mp4 -y")



def evaluate(
    model, 
    dataset: Dict[str, List[Tuple[torch.Tensor, Dict[str, int]]]],
    frame_rates: List[int],
    method: str,
    args,
    dmap_type: str = "threshold",
    dirty_thres: int = 30,
    dirty_topk: int = 100,
    sensi_expansion: int = 1,
    **kwargs: Any
):
    """
    Evaluate the model on the dataset at specified frame rates.
    """
    model.eval()
    model.counting()
    model.clear_counts()
    sequence_name = 0
    n_frames = 0

    pbar = tqdm(dataset, desc="Evaluating sequences", total=len(dataset))
    for sequence_data in pbar:
        # print(f"Evaluating sequence {sequence_name}/{len(dataset)} with {len(sequence_data)} frames")
        for frame_rate in frame_rates:
            # try:
            # print(f"Evaluating sequence: {sequence_name}, frame rate: {frame_rate} fps")

            evaluate_sequence(model, sequence_name, sequence_data, frame_rate, method, args, dmap_type, dirty_thres, dirty_topk, sensi_expansion, **kwargs)
            model.reset()
            n_frames += len(sequence_data)

            # except Exception as e:
            #     print(f"[Error] sequence {sequence_name}, frame rate {frame_rate} fps: {e}")
            #     break

        sequence_name += 1

    mean_ap = MeanAveragePrecision(box_format='xyxy')
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


@torch.no_grad()
def main(args):
    def tee_print(s, file, flush=True):
        print(s, flush=flush)
        print(s, file=file, flush=flush)

    def save_csv_results(results, output_dir, first_run=False):
        for key, val in results.items():
            with open(output_dir / f"{key}.csv", "w") as csv_file:
                if first_run:
                    print(dict_csv_header(val), file=csv_file)
                print(dict_csv_line(val), file=csv_file)

    def do_evaluation(title, results):
        with open(output_dir / "output.txt", "w") as tee_file:
            tee_print(title, tee_file)
            if isinstance(results, dict):
                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
                for key, val in results.items():
                    tee_print(key.capitalize(), tee_file)
                    tee_print(dict_string(val), tee_file)
            else:
                tee_print(results, tee_file)
            tee_print("", tee_file)
            tee_print(f"Recompute rate: {np.mean(recompute_rate):.4f}", tee_file)
            completed.append(title)

            save_path = output_dir / "pred_outputs.pt"
            cpu_outputs = []
            for d in outputs:
                cpu_outputs.append({
                    "boxes":  d["boxes"].cpu(),
                    "labels": d["labels"].cpu(),
                    "scores": d["scores"].cpu()
                })
            torch.save(cpu_outputs, save_path)
            print(f"Saved {len(cpu_outputs)} predictions to {save_path}")

    model, dataset, settings_dict = prepare_environment(args)

    results = evaluate(model, dataset, args.frame_rates, args.method, args, args.dmap_type, args.dirty_thres, args.dirty_topk, args.sensi_expansion, **settings_dict)

    completed = []
    model_name = f"{args.model}"
    frame_rate_str = f"{args.frame_rates[0]}fps"
    dirtiness_key = f"thres{args.dirty_thres}" if args.dmap_type == "threshold" else f"topk{args.dirty_topk}"
    output_dir = Path("output") / args.dataset / model_name / f"{args.method}_{frame_rate_str}" / dirtiness_key
    output_dir.mkdir(parents=True, exist_ok=True)

    do_evaluation("Vanilla", results)

    

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
    parser.add_argument("--dataset", type=str, default="imnet-vid", help="Dataset to evaluate on.",
        choices=["davis", "imnet-vid"],
    )
    parser.add_argument("--frame-rates", type=parse_int_list, default=[30], 
                       help="Frame rate(s) for evaluation. Comma-separated integers (e.g., 1,6,100).")
    parser.add_argument("--sequence", type=parse_str_list, default=None, 
                       help="Specific sequence(s) to evaluate on. Comma-separated strings (e.g., bear,camel). If None, evaluates on all sequences.")
    parser.add_argument("--dmap-type", type=str, choices=["threshold", "topk"], default="topk",
                       help="Type of dirtiness map to use. 'threshold' for thresholding, 'topk' for top-k dirtiness.")
    parser.add_argument("--dirty-thres", type=int, default=30, nargs="?",
                       help="Dirtiness threshold for the dirtiness map. Default is 30.")
    parser.add_argument("--dirty-topk", type=int, default=128, nargs="?",
                       help="Top-k dirtiness for the dirtiness map. Default is 100.")
    parser.add_argument("--sensi-expansion", type=int, default=1,
                       help="Expansion factor for the sensitivity map. Default is 1.")
    parser.add_argument("--method", type=str, choices=["ours", "evit", "maskvd", "stgt"], default="ours",
                       help="Method to use for evaluation. 'ours' for IPConv, 'evit' for Eventful ViT.")
    parser.add_argument("--device", type=str, default="cuda:0",)
    parser.add_argument("--roi-expand", type=int, default=1,
                       help="ROI expansion factor. Default is 1.")
    args = parser.parse_args()

    print(args)

    global_device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    main(args)

    
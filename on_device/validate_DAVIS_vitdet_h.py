import os
import cv2
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from torchvision.transforms import functional as F

from typing import Dict, Tuple, List

from ipconv.models import MaskedRCNN_ViT_H_FPN_Contexted
from ipconv.models.proc_image import visualize_detection, calculate_multi_iou
from ipconv.models.constants import COCO_LABELS_LIST
from ipconv.models.ViTDet.modeling.backbone.utils import expand_mask_neighbors, shrink_mask_neighbors

def create_sensitivity_map(
    boxes: List[List[float]],  # List of bounding boxes, each box is in a format of [x_min, y_min, x_max, y_max]
    scores: List[float],  # List of scores for each bounding box
) -> np.ndarray:
    # Create a blank sensitivity map
    sensitivity_map = np.zeros((1024, 1024), dtype=np.float32)

    # Iterate through each bounding box and its corresponding score
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = map(int, box)
        # Create a mask for the current bounding box
        mask = np.zeros((1024, 1024), dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = score
        # Add the mask to the sensitivity map
        sensitivity_map += mask

    # Expand the sensitivity map
    sensitivity_map = cv2.GaussianBlur(sensitivity_map, (63, 63), 1.5) * 255 * 255


    # Clip the sensitivity map to the range [0, 1]
    # sensitivity_map = np.clip(sensitivity_map, 0, 1)

    # Min-max normalization
    min_val = np.min(sensitivity_map)
    max_val = np.max(sensitivity_map)
    if max_val - min_val > 0:
        sensitivity_map = (sensitivity_map - min_val) / (max_val - min_val)
    else:
        sensitivity_map = np.zeros_like(sensitivity_map)

    return sensitivity_map

def create_dirtiness_map(
    anchor_image: np.ndarray, 
    current_image: np.ndarray,
    block_size: int = 16,
    dirty_thres: int = 30,
    chromakey: np.ndarray = np.array([123.675, 116.28, 103.53], dtype=np.uint8),
    sensi_map: np.ndarray = None,
) -> torch.Tensor:
    residual = cv2.absdiff(anchor_image, current_image)
    
    # inside current_image, if there is any pixel with chromakey color, set the residual as 0
    # chromakey_mask = np.all(current_image == chromakey, axis=-1)
    # residual[chromakey_mask] = 0

    dirtiness_map = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    image_H, image_W = residual.shape[:2]
    
    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
    if sensi_map is None:
        dirtiness_map = (dirtiness_map > dirty_thres).astype(np.float32)
    else:
        dirtiness_map = (dirtiness_map > dirty_thres * (1 - sensi_map)).astype(np.float32)

    dirtiness_map = cv2.GaussianBlur(dirtiness_map, (15, 15), 1.5)
    dirtiness_map = cv2.resize(dirtiness_map, (image_W // block_size, image_H // block_size), interpolation=cv2.INTER_LINEAR)
    dirtiness_map = (dirtiness_map > 0).astype(np.float32)

    dirtiness_map = torch.from_numpy(dirtiness_map).to("cuda")
    dirtiness_map = dirtiness_map.unsqueeze(0).unsqueeze(-1)

    # minimum recompute
    maxnum = 10
    while dirtiness_map.mean() < 0.01:

        dirtiness_map = expand_mask_neighbors(dirtiness_map)

        maxnum -= 1
        if maxnum == 0:
            break

    return dirtiness_map

def get_padded_image(image_ndarray: np.ndarray, size: Tuple[int, int], basic_scaling_factor: float = 1.05) -> np.ndarray:
    image_scaled = cv2.resize(image_ndarray, (int(image_ndarray.shape[1] * basic_scaling_factor), int(image_ndarray.shape[0] * basic_scaling_factor)), interpolation=cv2.INTER_LINEAR)

    shift_to_center = ((size[1] - image_scaled.shape[1]) // 2, (size[0] - image_scaled.shape[0]) // 2)

    padded_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    padded_image[:, :] = np.array([123.675, 116.28, 103.53], dtype=np.uint8)
    padded_image[shift_to_center[1]:shift_to_center[1] + image_scaled.shape[0], shift_to_center[0]:shift_to_center[0] + image_scaled.shape[1]] = image_scaled

    return padded_image

@torch.no_grad()
def estimate_affine_in_padded_anchor(
    anchor_padded_ndarray: np.ndarray,  # (1024, 1024, 3)
    target_ndarray: np.ndarray,         # (H, W, 3)
):
    # Find and match keypoints
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(target_ndarray, None)
    kp2, des2 = orb.detectAndCompute(anchor_padded_ndarray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select good matches
    good_matches = matches[:min(len(matches), 100)]

    # Extract coordinates of matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate Affine Transform
    # affine_matrix, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
    affine_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.LMEDS, maxIters=5000, confidence=0.999, refineIters=10)

    return affine_matrix

@torch.no_grad()
def apply_affine_and_pad(
    target_ndarray: np.ndarray,  # (H, W, 3)
    affine_matrix: np.ndarray,  # (2, 3)
) -> np.ndarray | None:
    result_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    H, W = target_ndarray.shape[:2]

    transformed_target = cv2.warpAffine(target_ndarray, affine_matrix, (1024, 1024))
    mask = transformed_target != 0

    # Check if any part of the transformed image is outside the 1024x1024 bounds
    points = np.array([[0, 0], [0, H], [W, 0], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.transform(points, affine_matrix)
    if np.any(transformed_points < 0) or np.any(transformed_points > 1024):
        return None

    result_image[:, :] = np.array([123.675, 116.28, 103.53], dtype=np.uint8)
    result_image[mask] = transformed_target[mask]

    return result_image

@torch.no_grad()
def affine_ground_truth_boxes(boxes_gt, affine_matrix):
    transformed_boxes = []
    for box in boxes_gt:
        x1, y1, x2, y2 = box

        point_lt = np.array([x1, y1], dtype=np.float32).reshape(-1, 1, 2)
        point_rt = np.array([x2, y1], dtype=np.float32).reshape(-1, 1, 2)
        point_lb = np.array([x1, y2], dtype=np.float32).reshape(-1, 1, 2)
        point_rb = np.array([x2, y2], dtype=np.float32).reshape(-1, 1, 2)

        src_pts = np.concatenate([point_lt, point_rt, point_lb, point_rb], axis=0)
        dst_pts = cv2.transform(src_pts, affine_matrix)

        x_min = int(np.mean(dst_pts[[0, 2], 0, 0]))
        y_min = int(np.mean(dst_pts[[0, 1], 0, 1]))
        x_max = int(np.mean(dst_pts[[1, 3], 0, 0]))
        y_max = int(np.mean(dst_pts[[2, 3], 0, 1]))

        transformed_boxes.append([x_min, y_min, x_max, y_max])
    return transformed_boxes


@torch.no_grad()
def single_inference(
    model: MaskedRCNN_ViT_H_FPN_Contexted,
    anchor_padded_ndarray: np.ndarray,  # (1024, 1024, 3)
    target_ndarray: np.ndarray,         # (H, W, 3)
    anchor_features: Dict[str, torch.Tensor],
    basic_scaling_factor: float = 1.05,
    recompute_threshold: float = 0.4,
    num_stages: int = 1,
    sensi_map: np.ndarray = None,
    pshift: int = 0,  # not used in this function, but can be used for future extensions
):
    # COMPRESSION PART must not included in latency
    comp_start = time.time()
    affine_matrix = estimate_affine_in_padded_anchor(anchor_padded_ndarray, target_ndarray)

    target_padded_ndarray = apply_affine_and_pad(target_ndarray, affine_matrix)

    # check if refresh is required
    refresh_anchor = False
    refresh_anchor |= (target_padded_ndarray is None)

    if not refresh_anchor:
        dirtiness_map = create_dirtiness_map(anchor_padded_ndarray, target_padded_ndarray)

        recompute_rate = np.mean(dirtiness_map.cpu().numpy())
        refresh_anchor |= (recompute_rate > recompute_threshold)

    # do jobs
    if refresh_anchor:
        target_padded_ndarray = get_padded_image(target_ndarray, (1024, 1024), basic_scaling_factor)
        comp_end = time.time()

        (boxes_cont, labels_cont, scores_cont), cached_features_dict = model.forward_contexted(target_padded_ndarray, pshift=pshift)
        
        # affine matrix: translation with shift_to_center
        target_scaled_ndarray = cv2.resize(target_ndarray, (int(target_ndarray.shape[1] * basic_scaling_factor), int(target_ndarray.shape[0] * basic_scaling_factor)), interpolation=cv2.INTER_LINEAR)
        shift_to_center = ((1024 - target_scaled_ndarray.shape[1]) // 2, (1024 - target_scaled_ndarray.shape[0]) // 2)
        affine_matrix = np.array([[basic_scaling_factor, 0, shift_to_center[0]], [0, basic_scaling_factor, shift_to_center[1]]], dtype=np.float32)

        return (boxes_cont, labels_cont, scores_cont), {
            "affine_matrix": affine_matrix,
            "target_padded_ndarray": target_padded_ndarray,
            "dirtiness_map": torch.ones((1, 64, 64, 1), dtype=torch.float32, device="cuda"),
            "cached_features_dict": cached_features_dict,
            "is_refreshed": refresh_anchor,
            "comp_time": comp_end - comp_start,
        }
    
    else:
        dirtiness_map = create_dirtiness_map(anchor_padded_ndarray, target_padded_ndarray, sensi_map=sensi_map)
        # dirtiness_map = torch.zeros_like(dirtiness_map, device="cuda")
        # dirtiness_map[0, 0, 0, 0] = 1.0

        num_stages = 1
        stage_map = dirtiness_map

        # stage map: expansion
        # stage_map = shrink_mask_neighbors(dirtiness_map)
        # stage_map += dirtiness_map

        # stage map: random
        # stage_map = torch.randint(1, num_stages + 1, dirtiness_map.shape, device="cuda")
        

        # stage map: sequential
        # stage_map = torch.zeros(dirtiness_map.shape, device="cuda")
        # for stage in range(1, num_stages + 1):
        #     start_width = int((stage - 1) * dirtiness_map.shape[2] / num_stages)
        #     end_width = int(stage * dirtiness_map.shape[2] / num_stages)
        #     stage_map[:, :, start_width:end_width, :] = stage
        comp_end = time.time()

        # staged inference
        cached_features_dict = anchor_features
        for stage in range(1, num_stages + 1):
            # create a mask for the current stage
            # mask = (stage_map == stage).float()
            # now_dmap = dirtiness_map * mask
            now_dmap = (stage_map == stage).float()

            now_dmap_comprate = torch.mean(now_dmap).item()

            if now_dmap_comprate == 0:
                continue

            is_last_stage = (stage == num_stages)
            (boxes_cont, labels_cont, scores_cont), cached_features_dict = model.forward_contexted(target_padded_ndarray, anchor_features=cached_features_dict, dirtiness_map=now_dmap, only_backbone=not is_last_stage, pshift=pshift)
            
        # (boxes_cont, labels_cont, scores_cont), cached_features_dict = model.forward_contexted(target_padded_ndarray, anchor_features=anchor_features, dirtiness_map=dirtiness_map)
        
        return (boxes_cont, labels_cont, scores_cont), {
            "affine_matrix": affine_matrix,
            "target_padded_ndarray": target_padded_ndarray,
            "dirtiness_map": dirtiness_map,
            "cached_features_dict": cached_features_dict,
            "is_refreshed": refresh_anchor,
            "comp_time": comp_end - comp_start,
        }
    

@torch.no_grad()
def validate_DAVIS(model, sequence_name, gop, data_root="/data/DAVIS", output_dir="./output/contexted_inference_vitdet_h", pshift: bool = False):
    # constants
    fixed_image_size = (1024, 1024)
    basic_scaling_factor = 1.05
    recompute_threshold = 0.4
    
    # load sequence
    sequence_path = f"{data_root}/JPEGImages/480p/{sequence_name}"
    image_names = sorted(os.listdir(sequence_path))

    annotations_path = os.path.join(data_root, "Annotations_bbox/480p", f"{sequence_name}.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    output_path = f"{output_dir}/{sequence_name}"
    os.makedirs(output_path, exist_ok=True)

    # iterate over images
    recompute_rates = []
    IoU_gt_results = []


    anchor_image_padded = None
    anchor_features = None
    sensi_map = None

    refresh_anchor = True

    start_time = time.time()
    uncounting_time = 0
    pbar = tqdm(enumerate(image_names), total=len(image_names), leave=True)
    for idx, iname in pbar:
        basename = os.path.splitext(iname)[0]

        # load ground truth
        annotation = annotations.get(basename, [])  # List of bounding boxes, each box is in a format of {'x_min': 431, 'y_min': 230, 'x_max': 460, 'y_max': 260, 'label': '14'}

        boxes_gt = [[float(box['x_min']), float(box['y_min']), float(box['x_max']), float(box['y_max'])] for box in annotation]
        labels_gt = [-1 for box in annotation]
        scores_gt = [1.0 for _ in annotation]

        load_start = time.time()
        current_image = cv2.imread(os.path.join(sequence_path, iname))
        load_end = time.time()
        uncounting_time += load_end - load_start
        
        # set scaling factor
        if current_image.shape[0] > fixed_image_size[0] or current_image.shape[1] > fixed_image_size[1]:
            scaling_factor = min(fixed_image_size[0] / current_image.shape[0], fixed_image_size[1] / current_image.shape[1])
        else:
            scaling_factor = min(basic_scaling_factor, fixed_image_size[0] / current_image.shape[0], fixed_image_size[1] / current_image.shape[1])

        if idx % gop == 0 or anchor_image_padded is None or refresh_anchor:
            # scale with basic_scaling_factor
            # check if the scaled image is bigger than fixed_image_size
            current_image = cv2.resize(current_image, (int(current_image.shape[1] * scaling_factor), int(current_image.shape[0] * scaling_factor)), interpolation=cv2.INTER_LINEAR)
            
            shift_to_center = ((fixed_image_size[1] - current_image.shape[1]) // 2, (fixed_image_size[0] - current_image.shape[0]) // 2)

            current_image_padded = np.zeros((1024, 1024, 3), dtype=np.uint8)
            current_image_padded[:, :] = np.array([123.675, 116.28, 103.53], dtype=np.uint8)
            current_image_padded[shift_to_center[1]:shift_to_center[1] + current_image.shape[0], shift_to_center[0]:shift_to_center[0] + current_image.shape[1]] = current_image

            (boxes_cont, labels_cont, scores_cont), cached_features_dict = model.forward_contexted(current_image_padded, pshift=pshift)
            
            # affine matrix: translation with shift_to_center and scale with scaling_factor
            affine_matrix = np.array([[scaling_factor, 0, shift_to_center[0]], [0, scaling_factor, shift_to_center[1]]], dtype=np.float32)
            dirtiness_map = torch.ones((1, 64, 64, 1), dtype=torch.float32, device="cuda")

            target_padded_ndarray = current_image_padded
            anchor_image_padded = current_image_padded
            anchor_features = cached_features_dict

            refresh_anchor = False

        else:
            num_stages = 1

            (boxes_cont, labels_cont, scores_cont), intermediate_dict = single_inference(
                model, 
                anchor_image_padded, 
                current_image, 
                anchor_features=anchor_features, 
                recompute_threshold=recompute_threshold,
                basic_scaling_factor=basic_scaling_factor,
                num_stages=num_stages,
                sensi_map=sensi_map,
                pshift=pshift,
            )

            affine_matrix = intermediate_dict["affine_matrix"]
            target_padded_ndarray = intermediate_dict["target_padded_ndarray"]
            dirtiness_map = intermediate_dict["dirtiness_map"]
            cached_features_dict = intermediate_dict["cached_features_dict"]
            is_refreshed = intermediate_dict["is_refreshed"]
            comp_time = intermediate_dict["comp_time"]

            uncounting_time += comp_time

            if not is_refreshed:
                # update padded anchor image
                dmap_resized = cv2.resize(dirtiness_map[0, :, :, 0].cpu().numpy(), (target_padded_ndarray.shape[1], target_padded_ndarray.shape[0]), interpolation=cv2.INTER_NEAREST)
                dmap_resized = np.stack([dmap_resized] * 3, axis=-1)
                new_anchor_padded_ndarray = anchor_image_padded * (1 - dmap_resized) + target_padded_ndarray * dmap_resized

                anchor_image_padded = new_anchor_padded_ndarray.astype(np.uint8)
                anchor_features = cached_features_dict
            else:
                # update anchor image
                anchor_image_padded = target_padded_ndarray
                anchor_features = cached_features_dict

        vis_start = time.time()
        # affine ground truth
        boxes_gt = affine_ground_truth_boxes(boxes_gt, affine_matrix)

        # stats
        IoU_gt = calculate_multi_iou(boxes_gt, labels_gt, boxes_cont, labels_cont)
        IoU_gt_mean = np.mean(IoU_gt)
        IoU_gt_results.append(IoU_gt_mean)
        
        scaling_factor = np.sqrt(np.linalg.det(affine_matrix[:2, :2]))
        if scaling_factor < 0.98:
            refresh_anchor = True

        recompute_rate = np.mean(dirtiness_map.cpu().numpy())
        recompute_rates.append(recompute_rate)

        # visualize
        dmap_resized = cv2.resize(dirtiness_map[0, :, :, 0].cpu().numpy(), (target_padded_ndarray.shape[1], target_padded_ndarray.shape[0]), interpolation=cv2.INTER_NEAREST)

        vis_image = target_padded_ndarray.copy()
        vis_image = vis_image.astype(np.uint16)
        vis_image[:, :, 1] = np.clip(vis_image[:, :, 1] + dmap_resized * 50, 0, 255)
        vis_image = vis_image.astype(np.uint8)

        vis_image = visualize_detection(vis_image, boxes_gt, labels_gt, scores_gt, threshold=0.5, colors=np.array([[0, 0, 255] for _ in range(len(COCO_LABELS_LIST))]), labels_list=model.COCO_LABELS_LIST)
        vis_image = visualize_detection(vis_image, boxes_cont, labels_cont, scores_cont, threshold=0.5, colors=np.array([[0, 255, 0] for _ in range(len(COCO_LABELS_LIST))]), labels_list=model.COCO_LABELS_LIST)

        sensi_map = create_sensitivity_map(boxes_cont, scores_cont)
        

        # write scaling factor at the left bottom corner
        cv2.putText(vis_image, f"Scale: {scaling_factor:.2f}", (10, 1014), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        os.makedirs(os.path.join(output_path, "temp"), exist_ok=True)
        cv2.imwrite(os.path.join(output_path, "temp", f"{idx:05d}.jpg"), vis_image)
        
        # visualize sensitivity map
        sensi_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        sensi_image = sensi_image.astype(np.uint16)
        sensi_image[:, :, 0] = np.clip(sensi_image[:, :, 0] + sensi_map * 255, 0, 255)
        sensi_image = sensi_image.astype(np.uint8)

        os.makedirs(os.path.join(output_path, "temp_sensi"), exist_ok=True)
        cv2.imwrite(os.path.join(output_path, "temp_sensi", f"{idx:05d}.jpg"), sensi_image)

        pbar.set_description(f"Recompute rate: {recompute_rate:.2f}, IoU (GT): {np.mean(IoU_gt):.2f}")

        vis_end = time.time()
        uncounting_time += vis_end - vis_start
    
    pbar.close()
    end_time = time.time()

    num_iters = len(image_names)
    elapsed_time = end_time - start_time - uncounting_time
    throughput = num_iters / elapsed_time
    
    # Make video of the results
    video_path = os.path.join(output_path, f"gop{gop}.mp4")
    os.system(f"ffmpeg -y -r 10 -i {output_path}/temp/%05d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p {video_path} > /dev/null 2>&1")
    # os.system(f"rm -rf {output_path}/temp")

    # avg_compute_rate, avg_iou_gt, avg_iou_full, inference_results
    avg_compute_rate = np.mean(recompute_rates)
    avg_iou_gt = np.mean(IoU_gt_results)

    return avg_compute_rate, avg_iou_gt, {
        "recompute_rates": recompute_rates,
        "IoU_gt_results": IoU_gt_results,
        "throughput": throughput,
    }


def main():

    data_root = "/data/DAVIS"
    output_dir = "./output/contexted_inference_vitdet_h"

    model = MaskedRCNN_ViT_H_FPN_Contexted("cuda")
    model.load_weight("./ipconv/models/model_final_7224f1.pkl")
    model.eval()

    # sequence_names = sorted(os.listdir("/data/DAVIS/JPEGImages/480p"))
    # if "bear_prep" in sequence_names:
    #     sequence_names.remove("bear_prep")
    # sequence_names = sequence_names[64:]
    # sequence_names = ["bear", "dog-gooses", "flamingo", "tuk-tuk", "skate-park"]
    sequence_names = ["bear"]
    # gops = [1, 2, 3, 6, 30, 100]
    # gops = [1, 2, 6, 100]
    gops = [100]

    log_text = "Sequence, "
    for gop in gops:
        log_text += f"gop{gop}_recompute_rate, "
    for gop in gops:
        log_text += f"gop{gop}_IoU_gt, "
    for gop in gops:
        log_text += f"gop{gop}_throughput, "
    print(log_text)

    # for pshift in [0, 1, 2, 4]:
    for pshift in [0]:
        print(f"Processing with pshift={pshift}")
        for sequence_name in sequence_names:
            recompute_rates = {}
            iou_gt_results = {}
            throughputs = {}

            os.makedirs(f"{output_dir}/{sequence_name}", exist_ok=True)

            for gop in gops:
                avg_compute_rate, avg_iou_gt, stat_dicts = validate_DAVIS(model, sequence_name, gop, data_root, output_dir, pshift=pshift)
                recompute_rates[gop] = stat_dicts["recompute_rates"]
                iou_gt_results[gop] = stat_dicts["IoU_gt_results"]
                throughputs[gop] = stat_dicts["throughput"]
            
            log_text = f"{sequence_name}, "
            for rrate in recompute_rates:
                log_text += f"{np.mean(recompute_rates[rrate]):f}, "
            for iou in iou_gt_results:
                log_text += f"{np.mean(iou_gt_results[iou]):f}, "
            for tput in throughputs:
                log_text += f"{1000/throughputs[tput]:f}, "

            print(log_text)

            # draw graphs
            # recompute rates
            plt.figure(figsize=(10, 5))
            plt.title(f"Image sequence: {sequence_name}", fontsize=20)  # Increased font size
            for gop in gops:
                plt.plot(recompute_rates[gop], label=f"GOP={gop}")
            plt.legend(fontsize=14)
            plt.xlabel("Frame", fontsize=16)
            plt.ylabel("Recompute rate", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid()
            plt.savefig(f"{output_dir}/{sequence_name}/recompute_rates.jpg")
            plt.close()

            # IoU results
            plt.figure(figsize=(10, 5))
            plt.title(f"Image sequence: {sequence_name}", fontsize=20)
            for gop in gops:
                plt.plot(iou_gt_results[gop], label=f"GOP={gop}")
            plt.legend(fontsize=14)
            plt.xlabel("Frame", fontsize=16)
            plt.ylabel("IoU (GT)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid()
            plt.savefig(f"{output_dir}/{sequence_name}/iou_results.jpg")
            plt.close()

            # throughput
            plt.figure(figsize=(10, 5))
            plt.title(f"Image sequence: {sequence_name}", fontsize=20)
            for gop in gops:
                plt.plot(throughputs[gop], label=f"GOP={gop}")
            plt.legend(fontsize=14)
            plt.xlabel("Frame", fontsize=16)
            plt.ylabel("Throughput (FPS)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid()
            plt.savefig(f"{output_dir}/{sequence_name}/throughput.jpg")
            plt.close()



if __name__ == "__main__":
    main()

import torch
import torchvision
from torch.nn import functional as F

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from typing import Tuple, Dict

from .constants import COCO_COLORS_ARRAY, COCO_LABELS_LIST


def apply_dirtiness_map(fname, feature, cache_features, dirtiness_map: torch.Tensor):
    if fname in cache_features:
        dirtiness_map_temp = torch.nn.functional.interpolate(dirtiness_map.clone(), size=feature.shape[-2:], mode='nearest')
        feature_new = cache_features[fname] * (1 - dirtiness_map_temp) + feature * dirtiness_map_temp

        feature = feature_new

    return feature, dirtiness_map


def expand_dirtiness_to_compute(dirtiness_map: torch.Tensor, kernel_size=6) -> torch.Tensor:
    # dirtiness_map: (B, 1, H, W)
    # we have to convolutional product with a kernel of size (kernel_size, kernel_size)
    # the kernel is a square matrix with all elements equal to 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=dirtiness_map.device)
    dirtiness_map_expanded = F.conv2d(dirtiness_map, kernel, padding=kernel_size // 2, stride=1, groups=1)
    dirtiness_map_expanded = (dirtiness_map_expanded > 0).float()

    return dirtiness_map_expanded

def create_compute_block(compute_map: torch.Tensor, block_size: int) -> torch.Tensor:
    # compute_map: (B, 1, H, W)
    # block_size: int
    # return: (B, 1, H, W)
    # split the compute_map into blocks of size (block_size, block_size)
    # if any element in the block is 1, the block is 1
    # otherwise, the block is 0
    B, C, H, W = compute_map.shape
    compute_map = compute_map.squeeze(1)
    compute_map = compute_map.reshape(B, H // block_size, block_size, W // block_size, block_size)
    compute_map = compute_map.max(dim=(2, 4)).values.unsqueeze(1)
    compute_map = F.interpolate(compute_map, size=(H, W), mode='nearest')

    return compute_map


def shift_by_float(tensor: torch.Tensor, tvec: tuple[float, float]) -> torch.Tensor:
    B, C, H, W = tensor.shape
    tx, ty = tvec
    
    # Normalize translation to [-1, 1] range (grid_sample expects normalized coordinates)
    tx /= W / 2
    ty /= H / 2
    
    # Construct affine transformation matrix for translation
    theta = torch.tensor([[1, 0, -tx], [0, 1, -ty]], dtype=torch.float, device=tensor.device)
    theta = theta.unsqueeze(0).expand(B, -1, -1)  # Expand for batch processing
    
    # Generate sampling grid
    grid = F.affine_grid(theta, [B, C, H, W], align_corners=True)
    
    # Perform grid sampling (bilinear interpolation)
    shifted_tensor = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return shifted_tensor


def shift_features_dict(
    features: Dict[str, torch.Tensor],
    ref_size: Tuple[int, int],
    shift_vector: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    shifted_features = {}
    for fname, feature in features.items():
        if feature is not None:
            scaled_shift_vector = (shift_vector[0] * feature.shape[-1] / ref_size[0],
                                   shift_vector[1] * feature.shape[-2] / ref_size[1])
            shifted_features[fname] = shift_by_float(feature, scaled_shift_vector)
        else:
            shifted_features[fname] = None

    return shifted_features


def refine_images(
    anchor_image_ndarray: np.ndarray,
    target_image_ndarray: np.ndarray,
) -> Tuple[np.ndarray | None, Tuple[float, float]]:

    gray1 = cv2.cvtColor(anchor_image_ndarray, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(target_image_ndarray, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if corners is None:
        return None, (0, 0)

    # Lucas-Kanade optical flow calculation
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, winSize=(15, 15), maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Select good points only
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = corners[st == 1]
        if good_old.shape[0] < 3:
            return None, (0, 0)

        # Outlier removal using RANSAC
        matrix, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
        inlier_old = good_old[inliers.flatten() == 1]
        inlier_new = good_new[inliers.flatten() == 1]

        # Calculate translation vector based on median (Translation Only)
        translation_vector = np.median(inlier_new - inlier_old, axis=0)
        # translation_vector = np.mean(inlier_new - inlier_old, axis=0)
        matrix = np.array([[1, 0, translation_vector[0]],
                           [0, 1, translation_vector[1]]], dtype=np.float32)

        # Apply image transformation
        aligned_image = cv2.warpAffine(anchor_image_ndarray, matrix, (anchor_image_ndarray.shape[1], anchor_image_ndarray.shape[0]))

        return aligned_image, tuple(translation_vector)
    else:
        return None, (0, 0)
    

def calculate_iou(target_box, infer_box):
    x1_gt, y1_gt, x2_gt, y2_gt = target_box
    x1, y1, x2, y2 = infer_box

    xA = max(x1, x1_gt)
    yA = max(y1, y1_gt)
    xB = min(x2, x2_gt)
    yB = min(y2, y2_gt)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def calculate_multi_iou(target_boxes, target_labels, infer_boxes, infer_labels):
    iou_results = []
    for target_box, target_label in zip(target_boxes, target_labels):
        iou_list = [calculate_iou(target_box, infer_box) for infer_box, infer_label in zip(infer_boxes, infer_labels) if target_label == infer_label or target_label == -1]
        iou_results.append(max(iou_list) if len(iou_list) > 0 else 0)

    return iou_results


def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.9,
    colors: np.ndarray = COCO_COLORS_ARRAY,
    labels_list: list[str] = COCO_LABELS_LIST,
) -> np.ndarray:
    for i in range(len(boxes)):
        if scores[i] > threshold:
            color = colors[labels[i]]
            x0, y0, x1, y1 = map(int, boxes[i])
            cv2.rectangle(image, (x0, y0), (x1, y1), (color * 255).astype(int).tolist(), 2)
            if labels[i] != -1:
                cv2.putText(
                    image,
                    f"{labels_list[labels[i]]}: {scores[i]:.2f}",
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (color * 255).astype(int).tolist(),
                    2,
                    cv2.LINE_AA,
                )

    return image


def graph_iou(IoU_gt_results, IoU_full_results, sequence_name, gop, output_path):
    avg_iou_gt = np.mean(IoU_gt_results)
    avg_iou_full = np.mean(IoU_full_results)

    plt.figure(figsize=(10, 5))
    plt.title(f"Image sequence: {sequence_name}, GOP: {gop}", fontsize=20)  # Increased font size
    plt.annotate(f"Average IoU (GT): {avg_iou_gt:.3f}", (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)  # Increased font size
    plt.annotate(f"Average IoU (full): {avg_iou_full:.3f}", (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)  # Increased font size
    plt.plot(IoU_gt_results, label="IoU (GT)")
    plt.plot(IoU_full_results, label="IoU (full)")
    plt.legend(fontsize=14)  # Increased font size
    plt.xlabel("Frame", fontsize=16)  # Increased font size
    plt.ylabel("IoU", fontsize=16)  # Increased font size
    plt.xticks(fontsize=14)  # Set x-axis tick label font size
    plt.yticks(fontsize=14)  # Set y-axis tick label font size
    plt.grid()
    plt.savefig(os.path.join(output_path, f"gop{gop}_iou.jpg"))
    plt.close()


def graph_recompute(compute_rates, sequence_name, gop, output_path):
    avg_compute_rate = np.mean(compute_rates)

    plt.figure(figsize=(10, 5))
    plt.title(f"Image sequence: {sequence_name}, GOP: {gop}", fontsize=20)  # Increased font size
    plt.annotate(f"Average recompute rate: {avg_compute_rate:.3f}", (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=14)  # Increased font size
    plt.plot(compute_rates, label="Recompute rate")
    plt.legend(fontsize=14)  # Increased font size
    plt.xlabel("Frame", fontsize=16)  # Increased font size
    plt.ylabel("Recompute rate", fontsize=16)  # Increased font size
    plt.xticks(fontsize=14)  # Set x-axis tick label font size
    plt.yticks(fontsize=14)  # Set y-axis tick label font size
    plt.grid()
    plt.savefig(os.path.join(output_path, f"gop{gop}_recompute.jpg"))
    plt.close()

if __name__ == "__main__":
    import os

    # DEMO 1: Test refine_images
    anchor_image = cv2.imread('/data/DAVIS/JPEGImages/480p/bear/00000.jpg')
    target_image = cv2.imread('/data/DAVIS/JPEGImages/480p/bear/00001.jpg')

    refined_image, translation_vector = refine_images(anchor_image, target_image)
    print(translation_vector)

    residual = cv2.absdiff(target_image, refined_image)
    residual = cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)

    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/residual.jpg', residual)
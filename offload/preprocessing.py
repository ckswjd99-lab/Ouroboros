import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math

from typing import List, Tuple, Dict


def create_sensitivity_map(
    boxes: List[List[float]],
    scores: List[float],
    map_size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """
    Create a sensitivity map based on bounding boxes and scores.

    Arguments:
        boxes (List[List[float]]): List of bounding boxes, each box is in a format of [x_min, y_min, x_max, y_max].
        scores (List[float]): List of scores corresponding to each bounding box.

    Returns:
        np.ndarray: Sensitivity map of shape (64, 64).
    """
    # Create a blank sensitivity map
    sensitivity_map = np.zeros(map_size, dtype=np.float32)

    # Iterate through each bounding box and its corresponding score
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = map(int, box)
        # Create a mask for the current bounding box
        mask = np.zeros(map_size, dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = score
        # Add the mask to the sensitivity map
        sensitivity_map += mask

    # Expand the sensitivity map
    sensitivity_map = cv2.GaussianBlur(sensitivity_map, (63, 63), 1.5) * 255 * 255

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
    dirty_thres: int = 20,
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
    # maxnum = 10
    # while dirtiness_map.mean() < 0.01:
    #     dirtiness_map = expand_mask_neighbors(dirtiness_map)

    #     maxnum -= 1
    #     if maxnum == 0:
    #         break

    return dirtiness_map

def estimate_affine_in_padded_anchor(
    anchor_padded_ndarray: np.ndarray,  # (1024, 1024, 3)
    target_ndarray: np.ndarray,         # (H, W, 3)
) -> np.ndarray:
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

def estimate_affine_in_padded_anchor_fast(
    anchor_padded_ndarray: np.ndarray,  # (1024, 1024, 3)
    target_ndarray: np.ndarray,         # (H, W, 3)
) -> np.ndarray:
    scaler = 2

    # Resize images into a half
    anchor_padded_ndarray = cv2.resize(anchor_padded_ndarray, (anchor_padded_ndarray.shape[1] // scaler, anchor_padded_ndarray.shape[0] // scaler), interpolation=cv2.INTER_LINEAR)
    target_ndarray = cv2.resize(target_ndarray, (target_ndarray.shape[1] // scaler, target_ndarray.shape[0] // scaler), interpolation=cv2.INTER_LINEAR)

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

    # Restore the scale
    affine_matrix[:, 2] *= scaler

    return affine_matrix


def estimate_translation_in_padded_anchor(
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

    # Calculate Translation Transform
    translation_matrix = np.mean(dst_pts - src_pts, axis=0).reshape(1, 2)

    return translation_matrix

def pattern_match_in_padded_anchor(
    anchor_padded_ndarray: np.ndarray,  # (1024, 1024, 3)
    target_ndarray: np.ndarray,  # (H, W, 3)
) -> np.ndarray | None:    
    H, W = target_ndarray.shape[:2]

    # Perform template matching
    result = cv2.matchTemplate(anchor_padded_ndarray, target_ndarray, cv2.TM_SQDIFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Check if the match is good enough
    if result.max() < 0.5:
        return None

    # Calculate the translation vector
    translation_vector = np.array(max_loc) - np.array([W // 2, H // 2])

    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, translation_vector[0]], [0, 1, translation_vector[1]]])

    return translation_matrix

def shift_anchor_features(anchor_features: dict, shift_x: int, shift_y: int):
    """
    anchor_features의 모든 qkv/out 텐서를 shift_x, shift_y만큼 블록 단위로 이동시킴.
    """
    for key, value in anchor_features.items():
        if "qkv" in key:
            num_windows = value.shape[1]
            num_hw = value.shape[3]

            sqrt_num_windows = int(math.sqrt(num_windows))
            sqrt_num_hw = int(math.sqrt(num_hw))

            key_reshaped = value.view(
                value.shape[0], sqrt_num_windows, sqrt_num_windows, value.shape[2],
                sqrt_num_hw, sqrt_num_hw, value.shape[4]
            )
            key_reshaped = key_reshaped.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
                value.shape[0], sqrt_num_windows * sqrt_num_hw, sqrt_num_windows * sqrt_num_hw, value.shape[2], value.shape[4]
            )
            key_reshaped = key_reshaped.roll(shifts=(-shift_y, -shift_x), dims=(1, 2))
            key_reshaped = key_reshaped.view(
                value.shape[0], sqrt_num_windows, sqrt_num_hw, sqrt_num_windows, sqrt_num_hw, value.shape[2], value.shape[4]
            )
            key_reshaped = key_reshaped.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
            key_reshaped = key_reshaped.view(*value.shape)
            anchor_features[key] = key_reshaped
        if "out" in key:
            # value: (B, H, W, C)
            value = value.roll(shifts=(-shift_y, -shift_x), dims=(1, 2))
            anchor_features[key] = value
    
    return anchor_features

def apply_affine_and_pad(
    target_ndarray: np.ndarray,  # (H, W, 3)
    affine_matrix: np.ndarray,  # (2, 3)
    block_size: int = 16,
    filling_color: list = [123.675, 116.28, 103.53]
) -> np.ndarray | None:
    result_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    H, W = target_ndarray.shape[:2]

    points = np.array([[0, 0], [0, H], [W, 0], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.transform(points, affine_matrix)

    shift_x, shift_y = 0, 0
    if np.any(transformed_points < 0) or np.any(transformed_points > 1024):
        shift_x_minus = math.floor(min(0, transformed_points[:, 0, 0].min() / block_size))
        shift_x_plus = math.ceil(max(0, (transformed_points[:, 0, 0].max() - 1024) / block_size))
        shift_y_minus = math.floor(min(0, transformed_points[:, 0, 1].min() / block_size))
        shift_y_plus = math.ceil(max(0, (transformed_points[:, 0, 1].max() - 1024) / block_size))
        shift_x = int(shift_x_minus + shift_x_plus)
        shift_y = int(shift_y_minus + shift_y_plus)

        affine_matrix[0, 2] -= shift_x * block_size
        affine_matrix[1, 2] -= shift_y * block_size

    transformed_target = cv2.warpAffine(target_ndarray, affine_matrix, (1024, 1024))
    mask = transformed_target != 0

    result_image[:, :] = np.array(filling_color, dtype=np.uint8)
    result_image[mask] = transformed_target[mask]

    return result_image, (shift_x * block_size, shift_y * block_size), affine_matrix

def affine_ground_truth_boxes(
    boxes_gt: List[List[float]],  # List of bounding boxes, each box is in a format of [x_min, y_min, x_max, y_max]
    affine_matrix: np.ndarray,  # (2, 3)
):
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

def get_padded_image(
    image_ndarray: np.ndarray, 
    size: Tuple[int, int], 
    basic_scaling_factor: float = 1.0,
    filling_color: List[float] = [123.675, 116.28, 103.53]
) -> np.ndarray:
    image_scaled = cv2.resize(image_ndarray, (int(image_ndarray.shape[1] * basic_scaling_factor), int(image_ndarray.shape[0] * basic_scaling_factor)), interpolation=cv2.INTER_LINEAR)

    shift_to_center = ((size[1] - image_scaled.shape[1]) // 2, (size[0] - image_scaled.shape[0]) // 2)

    padded_image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    padded_image[:, :] = np.array(filling_color, dtype=np.uint8)
    padded_image[shift_to_center[1]:shift_to_center[1] + image_scaled.shape[0], shift_to_center[0]:shift_to_center[0] + image_scaled.shape[1]] = image_scaled

    return padded_image

def expand_mask_neighbors(mask_4d: torch.Tensor) -> torch.Tensor:
    mask_4d = mask_4d.permute(0, 3, 1, 2)  # (1, 1, 64, 64)
    kernel = torch.ones((1, 1, 3, 3), device=mask_4d.device, dtype=mask_4d.dtype)
    
    expanded = F.conv2d(mask_4d, kernel, padding=1)
    expanded = (expanded > 0).float()
    expanded = expanded.permute(0, 2, 3, 1)
    
    return expanded

def shrink_mask_neighbors(mask_4d: torch.Tensor) -> torch.Tensor:
    mask_4d = mask_4d.permute(0, 3, 1, 2)  # (1, 1, 64, 64)
    kernel = torch.ones((1, 1, 3, 3), device=mask_4d.device, dtype=mask_4d.dtype)
    
    shrunk = F.conv2d(mask_4d, kernel, padding=1)
    shrunk = (shrunk == 9).float()
    shrunk = shrunk.permute(0, 2, 3, 1)
    
    return shrunk

def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """
    Convert a numpy ndarray to bytes.

    Args:
        ndarray (np.ndarray): The numpy array to convert.

    Returns:
        bytes: The byte representation of the array.
    """

    shape = ndarray.shape
    dtype = ndarray.dtype

    # Convert the array to bytes
    bytes_data = ndarray.tobytes()

    # Create a header with shape and dtype
    header = f"{len(shape)}|{'|'.join(map(str, shape))}|{dtype.name}|".encode()
    len_header = len(header).to_bytes(4, 'big')

    # Combine header and data
    return len_header + header + bytes_data

def bytes_to_ndarray(data: bytes) -> np.ndarray:
    """
    Convert bytes back to a numpy ndarray.

    Args:
        data (bytes): The byte data to convert.

    Returns:
        np.ndarray: The reconstructed numpy array.
    """

    # Read the header length
    len_header = int.from_bytes(data[:4], 'big')
    
    # Extract the header
    header = data[4:4 + len_header].decode()
    
    # Parse the header
    parts = header.split('|')
    num_dims = int(parts[0])
    shape = tuple(map(int, parts[1:num_dims + 1]))
    dtype = np.dtype(parts[num_dims + 1])

    # Extract the array data
    array_data = data[4 + len_header:]

    # Create the ndarray from the bytes
    return np.frombuffer(array_data, dtype=dtype).reshape(shape)
    

def load_video(video_path: str) -> List[np.ndarray]:
    """
    Load a video file and return its frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List[np.ndarray]: List of frames from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames
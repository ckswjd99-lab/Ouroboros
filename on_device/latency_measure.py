import numpy as np
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any, Tuple

from ipconv.models import (
    MaskedRCNN_ViT_B_FPN_Contexted, MaskedRCNN_ViT_L_FPN_Contexted, MaskedRCNN_ViT_H_FPN_Contexted,
    CascadeMaskRCNN_Swin_B_Contexted, CascadeMaskRCNN_Swin_L_Contexted,
    # CascadeMaskRCNN_MViT_B_Contexted
)

def measure_latency_memory(
    model: nn.Module,
    patch_keep_rate: float,
    method: str,
    input_size: int
):
    model.eval()
    dummy_input = np.zeros((input_size, input_size, 3))
    block_size = 16
    num_blocks_sqrt = input_size // block_size


    dmap = torch.zeros((1, num_blocks_sqrt, num_blocks_sqrt, 1), dtype=torch.float32, device="cuda")
    num_patches = num_blocks_sqrt * num_blocks_sqrt
    num_keep = int(num_patches * patch_keep_rate)
    idx_rand = torch.randperm(num_patches)[:num_keep]
    dmap.view(-1)[idx_rand] = 1.0

    num_warmup = 3
    num_repeats = 5

    # inference_func = model.forward_contexted if method == "ours" else model.forward_eventful
    if method == "vanilla":
        inference_func = model.forward
    elif method == "ours":
        inference_func = model.forward_contexted
    elif method == "eventful":
        inference_func = model.forward_eventful
    elif method == "maskvd":
        inference_func = model.forward_maskvd
    elif method == "stgt":
        inference_func = model.forward_stgt
    else:
        raise ValueError(f"Unknown method: {method}")

    for _ in range(num_warmup):
        output = inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True)

    model.counting()
    model.clear_counts()

    start_time = time.time()
    for _ in tqdm(range(num_repeats), leave=False):
        inference_func(dummy_input, dirtiness_map=dmap, only_backbone=True, anchor_features=output[1])
    end_time = time.time()

    cache_size = 0

    if method != "vanilla":
        cache = output[1]

        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                size = value.element_size() * value.numel()
                cache_size += size
    
    counts = model.total_counts() / num_repeats
    model.clear_counts()

    latency = (end_time - start_time) / num_repeats
    return latency, cache_size, counts


@torch.no_grad()
def main():
    models_dict = {
        # "ViT-base": MaskedRCNN_ViT_B_FPN_Contexted,
        "ViT-large": MaskedRCNN_ViT_L_FPN_Contexted,
        # "ViT-huge": MaskedRCNN_ViT_H_FPN_Contexted,
        # "Swin-base": CascadeMaskRCNN_Swin_B_Contexted,
        # "Swin-large": CascadeMaskRCNN_Swin_L_Contexted,
        # "MViT-B": CascadeMaskRCNN_MViT_B_Contexted,
    }

    keep_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # keep_rates = [0.1]
    # keep_rates = [0.0776, 0.1154, 0.1934, 0.3731]
    # keep_rates = [0.0724, 0.1082, 0.1764, 0.3079]
    # keep_rates = [0.2698, 0.2659]

    # methods = ["ours", "eventful", "maskvd", "stgt"]
    methods = ["ours", "stgt"]
    # methods = ["vanilla"]
    # methods = ["ours"]
    # methods = ["stgt"]

    input_sizes = [1024]

    for input_size in input_sizes:
        for mname, model_class in models_dict.items():
            model = model_class("cuda")
            model.eval()

            for method in methods:
                for keep_rate in keep_rates:
                    latency, cache_size, num_count = measure_latency_memory(model, keep_rate, method, input_size)
                    # print(f"Model: {mname}, Input: {input_size}, Method: {method}, Patch Keep Rate: {keep_rate}, Latency: {latency:.4f} seconds, Cache Size: {cache_size / (1024 * 1024):.2f} MB, ")
                    # print(sum(num_count.values()))
                    print(f"{input_size},{mname},{method},{keep_rate},{latency:.4f},{sum(num_count.values())},{cache_size / (1024 * 1024):.2f}")

if __name__ == "__main__":
    main()
    print("Latency measurement completed.")
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "window_partition",
    "window_unpartition",
    "add_decomposed_rel_pos",
    "get_abs_pos",
    "PatchEmbed",
]


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size

    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # (left, right, top, bottom, ...)
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)  # [B, num_h, num_w, window_size, window_size, C]

    windows = x.reshape(-1, window_size, window_size, C)

    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = 2 * max(q_size, k_size) - 1
    
    # Interpolate rel_pos only if necessary
    if rel_pos.shape[0] != max_rel_dist:
        # rel_pos (L, C) -> (1, C, L) -> interpolate -> (1, C, max_rel_dist) -> (max_rel_dist, C)
        rel_pos = rel_pos.unsqueeze(0).permute(0, 2, 1)  # (1, C, L)
        rel_pos = F.interpolate(
            rel_pos,
            size=max_rel_dist,
            mode="linear",
            align_corners=False
        )
        rel_pos_resized = rel_pos.permute(0, 2, 1).squeeze(0)  # (max_rel_dist, C)
    else:
        rel_pos_resized = rel_pos

    # Compute relative coords
    #    Possible caching if q_size, k_size are repeated
    scale_factor = max(k_size / q_size, 1.0)
    q_coords = torch.arange(q_size, dtype=torch.float32, device=rel_pos.device).mul_(scale_factor)
    
    scale_factor2 = max(q_size / k_size, 1.0)
    k_coords = torch.arange(k_size, dtype=torch.float32, device=rel_pos.device).mul_(scale_factor2)

    # int indexing => shift by (k_size - 1)*scale_factor2, then round
    relative_coords = (q_coords[:, None] - k_coords[None, :]) + (k_size - 1) * scale_factor2

    # Index into rel_pos_resized
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size, dmap=None):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    
    # get_rel_pos optimized
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)  # shape [q_h*k_h or max_rel_dist, C]
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)  # shape [q_w*k_w or max_rel_dist, C]

    B, _, dim = q.shape
    # q: (B, q_h * q_w, dim) -> (B, q_h, q_w, dim)
    q_4d = q.view(B, q_h, q_w, dim)

    # rel_h: (B, q_h, q_w, k_h)
    # rel_w: (B, q_h, q_w, k_w)
    rel_h = torch.einsum("bhwc,hkc->bhwk", q_4d, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", q_4d, Rw)
    
    # attn shape: (B, q_h*q_w, k_h*k_w)
    if dmap is None:
        attn_4d = attn.view(B, q_h * q_w, k_h, k_w)
        rel_h = rel_h.reshape(B, q_h * q_w, k_h)
        rel_w = rel_w.reshape(B, q_h * q_w, k_w)
        attn_4d = attn_4d + rel_h[:, :, :, None] + rel_w[:, :, None, :]

        # flatten back
        attn = attn_4d.view(B, q_h * q_w, k_h * k_w)
    else:
        dmap_flat = dmap.view(-1)
        dmap = dmap.reshape(q_h, q_w)

        attn_4d = attn.reshape(B, q_h * q_w, k_h, k_w)
        rel_h = rel_h.reshape(B, q_h * q_w, k_h)
        rel_w = rel_w.reshape(B, q_h * q_w, k_w)

        attn_4d_sel = attn_4d[:, dmap_flat == 1, :, :]
        rel_h_sel = rel_h[:, dmap_flat == 1, :]
        rel_w_sel = rel_w[:, dmap_flat == 1, :]
        attn_4d_sel = attn_4d_sel + rel_h_sel[:, :, :, None] + rel_w_sel[:, :, None, :]
        attn_4d_sel = attn_4d_sel.view(B, -1, k_h * k_w)

        # broadcast
        attn = torch.zeros(
            B, q_h * q_w, k_h * k_w, device=attn.device, dtype=attn.dtype
        )
        attn[:, dmap_flat == 1, :] = attn_4d_sel

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def partial_mlp_inference(x, dmap, mlp_module, drop_path_fn=None):
    """
    x: (B, H, W, C)
    dmap: (B, H, W, 1), with values 0 or 1
    mlp_module: an MLP (FFN) module for f(x) (e.g., block.mlp)
    drop_path_fn: optional drop_path function (e.g., block.drop_path)

    Returns:
        (B, H, W, C) where only positions with dmap=1 are updated by the MLP.
    """
    B, H, W, C = x.shape
    
    x_flat = x.view(-1, C)  # shape: (N, C), where N = B*H*W

    dmap_flat = dmap.view(-1)  # shape: (N,)

    dirty_indices = torch.nonzero(dmap_flat, as_tuple=True)[0]  # shape: (D,)
    if dirty_indices.numel() == 0:
        return x

    dirty_tokens = x_flat[dirty_indices, :]  # shape: (D, C)

    updated_tokens = mlp_module(dirty_tokens)

    if drop_path_fn is not None:
        updated_tokens = drop_path_fn(updated_tokens)

    x_flat[dirty_indices, :] = updated_tokens
    x_updated = x_flat.view(B, H, W, C)

    return x_updated



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
import torch
import torch.nn as nn
import numpy as np

from .modeling.backbone.utils import window_partition, window_unpartition, add_decomposed_rel_pos, window_reverse, AddDecomposedRelPos
from .modeling.meta_arch import GeneralizedRCNN

from .eventful_transformer.base import ExtendedModule
from .eventful_transformer.counting import CountedAdd, CountedMatmul

from .structures import ImageList

import pickle
import torch.nn.functional as F

from typing import Dict

class CascadeMaskRCNN_Swin_Contexted(ExtendedModule):
    def __init__(self, num_classes=80, device="cuda"):
        super(CascadeMaskRCNN_Swin_Contexted, self).__init__()
        self.base_model = nn.Identity()
        self.num_classes = num_classes
        self.device = device

        self.COCO_LABELS_LIST = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.add = CountedAdd()
        self.matmul = CountedMatmul()
        self.add_decomposed_rel_pos = AddDecomposedRelPos()

    def forward(self, image_ndarray: np.ndarray, **kwargs):
        only_backbone = kwargs.get("only_backbone", False)

        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=self.device).permute(2, 0, 1)
        input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
        
        # detections = self.base_model(input)
        images = [self.base_model._move_to_current_device(x["image"]) for x in input]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )

        features = self.base_model.backbone(images.tensor)
        
        if only_backbone:
            return ([], [], []), {}

        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        detections = self._postprocess(results, input, images.image_sizes)

        predictions = detections[0]
        boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions["instances"].pred_classes.cpu().numpy()
        scores = predictions["instances"].scores.cpu().numpy()

        return boxes, labels, scores

    def forward_analyzed(self, image_ndarray: np.ndarray):
        return self.forward(image_ndarray)
    
    def forward_contexted(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cache_feature = {}

        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device).half()
        batched_inputs = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        # Preprocess image
        base_model = self.base_model
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )
        
        # Backbone
        backbone = base_model.backbone
        swin_model = backbone.bottom_up
        x = images.tensor
        
        x = swin_model.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        ## projection without APE
        x = x.flatten(2).transpose(1, 2)
        x = swin_model.pos_drop(x)

        outs = {}
        for lidx in range(swin_model.num_layers):
            # Swin Transformer Layer
            layer = swin_model.layers[lidx]
            LH, LW = Wh, Ww
            
            ## downsample the dirtiness map to the current layer's feature map size
            dmap_layer = dirtiness_map.clone().squeeze(-1).unsqueeze(0)  # 1, 1, LH, LW
            dmap_layer = F.interpolate(dmap_layer, size=(LH, LW), mode="area")
            dmap_layer = (dmap_layer > 0).float()
            dmap_layer = dmap_layer.unsqueeze(-1).squeeze(0)  # 1, LH, LW, 1

            Hp = int(np.ceil(LH / layer.window_size)) * layer.window_size
            Wp = int(np.ceil(LW / layer.window_size)) * layer.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
            h_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            w_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(
                img_mask, layer.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, layer.window_size * layer.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )

            for bidx, block in enumerate(layer.blocks):
                # Swin Transformer Block
                block.H, block.W = LH, LW
                Block_B, Block_L, Block_C = x.shape
                Block_H, Block_W = block.H, block.W

                shortcut = x
                x = block.norm1(x)
                x = x.view(Block_B, Block_H, Block_W, Block_C)

                # pad feature maps to multiples of window size
                pad_l = pad_t = 0
                pad_r = (block.window_size - Block_W % block.window_size) % block.window_size
                pad_b = (block.window_size - Block_H % block.window_size) % block.window_size
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                _, Hp, Wp, _ = x.shape

                # cyclic shift
                if block.shift_size > 0:
                    shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                    attn_mask = attn_mask
                else:
                    shifted_x = x
                    attn_mask = None

                # partition windows
                x_windows, _ = window_partition(shifted_x, block.window_size)
                dmap_windows, _ = window_partition(dmap_layer, block.window_size)
                # nW*B, window_size, window_size, C
                
                x_windows = x_windows.view(-1, block.window_size * block.window_size, Block_C)
                # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                ATTN_B_, ATTN_N, ATTN_C = x_windows.shape

                qkv = block.attn.qkv(x_windows) # (ATTN_B_, ATTN_N, 3 * ATTN_C)
                qkv = qkv.reshape(ATTN_B_, ATTN_N, 3, block.attn.num_heads, ATTN_C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # q: Tensor(ATTN_B_, block.attn.num_heads, ATTN_N, ATTN_C // block.attn.num_heads)
                q = q * block.attn.scale
                attn = self.matmul(q, k.transpose(-2, -1))

                relative_position_bias = block.attn.relative_position_bias_table[
                    block.attn.relative_position_index.view(-1)
                ].view(
                    block.attn.window_size[0] * block.attn.window_size[1], block.attn.window_size[0] * block.attn.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = self.add(attn, relative_position_bias.unsqueeze(0))

                if attn_mask is not None:
                    nW = attn_mask.shape[0]
                    attn = attn.view(ATTN_B_ // nW, nW, block.attn.num_heads, ATTN_N, ATTN_N) + attn_mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, block.attn.num_heads, ATTN_N, ATTN_N)
                    attn = block.attn.softmax(attn)
                else:
                    attn = block.attn.softmax(attn)

                attn = block.attn.attn_drop(attn)

                x_windows = self.matmul(attn, v).transpose(1, 2).reshape(ATTN_B_, ATTN_N, ATTN_C)
                x_windows = block.attn.proj(x_windows)
                x_windows = block.attn.proj_drop(x_windows)

                attn_windows = x_windows

                # merge windows
                attn_windows = attn_windows.view(-1, block.window_size, block.window_size, Block_C)
                shifted_x = window_reverse(attn_windows, block.window_size, Hp, Wp)  # B H' W' C

                # reverse cyclic shift
                if block.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                else:
                    x = shifted_x

                if pad_r > 0 or pad_b > 0:
                    x = x[:, :Block_H, :Block_W, :].contiguous()

                x = x.view(Block_B, Block_H * Block_W, Block_C)

                # FFN
                x = self.add(shortcut, block.drop_path(x))

                dindice_for_embeddings = torch.nonzero(dmap_layer.view(-1) == 1, as_tuple=False).view(-1)
                x_sel = x.reshape(-1, Block_C)
                x_sel = F.embedding(dindice_for_embeddings, x_sel)
                mlp_x_selected = block.drop_path(block.mlp(block.norm2(x_sel)))
                x_sel = self.add(x_sel, mlp_x_selected)
                
                fname = f"layer{lidx}_block{bidx}_ffn_out"
                if fname in anchor_features:
                    x_cached = anchor_features[fname].clone()
                    x_cached[dindice_for_embeddings] = x_sel
                    x = x_cached
                else:
                    x_cached = x.view(-1, Block_C).clone()
                    x_cached[dindice_for_embeddings] = x_sel
                    x = x_cached
                new_cache_feature[fname] = x.clone()
                x = x.unsqueeze(0)
                
            
            if layer.downsample is not None:
                x_down = layer.downsample(x, LH, LW)
                Wh, Ww = (LH + 1) // 2, (LW + 1) // 2
                x_out, H, W, x, Wh, Ww = x, LH, LW, x_down, Wh, Ww
            else:
                x_out, H, W, x, Wh, Ww = x, LH, LW, x, LH, LW

            if lidx in swin_model.out_indices:
                norm_layer = getattr(swin_model, f"norm{lidx}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, swin_model.num_features[lidx]).permute(0, 3, 1, 2).contiguous()
                outs["p{}".format(lidx)] = out

        bottom_up_features = outs

        if only_backbone:
            return ([], [], []), new_cache_feature, []

        # FPN
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        ## reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            ## Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            ## Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        
        features = {f: res for f, res in zip(backbone._out_features, results)}
        
        # Post-process features
        proposals, _ = base_model.proposal_generator(images, features, None)
        results, _ = base_model.roi_heads(images, features, proposals, None)

        predictions = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # Process predictions
        boxes = predictions[0]["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions[0]["instances"].pred_classes.cpu().numpy()
        scores = predictions[0]["instances"].scores.cpu().numpy()
        pred_masks = predictions[0]["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks


    def forward_eventful(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cache_feature = {}

        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device).half()
        batched_inputs = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        # Preprocess image
        base_model = self.base_model
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )
        
        # Backbone
        backbone = base_model.backbone
        swin_model = backbone.bottom_up
        x = images.tensor
        
        x = swin_model.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        ## projection without APE
        x = x.flatten(2).transpose(1, 2)
        x = swin_model.pos_drop(x)

        outs = {}
        for lidx in range(swin_model.num_layers):
            # Swin Transformer Layer
            layer = swin_model.layers[lidx]
            LH, LW = Wh, Ww
            
            ## downsample the dirtiness map to the current layer's feature map size
            dmap_layer = dirtiness_map.clone().squeeze(-1).unsqueeze(0)  # 1, 1, LH, LW
            dmap_layer = F.interpolate(dmap_layer, size=(LH, LW), mode="area")
            dmap_layer = (dmap_layer > 0).float()
            dmap_layer = dmap_layer.unsqueeze(-1).squeeze(0)  # 1, LH, LW, 1

            Hp = int(np.ceil(LH / layer.window_size)) * layer.window_size
            Wp = int(np.ceil(LW / layer.window_size)) * layer.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
            h_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            w_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(
                img_mask, layer.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, layer.window_size * layer.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )

            for bidx, block in enumerate(layer.blocks):
                # Swin Transformer Block
                block.H, block.W = LH, LW
                Block_B, Block_L, Block_C = x.shape
                Block_H, Block_W = block.H, block.W

                shortcut = x
                x = block.norm1(x)
                x = x.view(Block_B, Block_H, Block_W, Block_C)

                # pad feature maps to multiples of window size
                pad_l = pad_t = 0
                pad_r = (block.window_size - Block_W % block.window_size) % block.window_size
                pad_b = (block.window_size - Block_H % block.window_size) % block.window_size
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                _, Hp, Wp, _ = x.shape

                # cyclic shift
                if block.shift_size > 0:
                    shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                    attn_mask = attn_mask
                else:
                    shifted_x = x
                    attn_mask = None

                # partition windows
                x_windows, _ = window_partition(shifted_x, block.window_size)
                dmap_windows, _ = window_partition(dmap_layer, block.window_size)
                # nW*B, window_size, window_size, C
                
                x_windows = x_windows.view(-1, block.window_size * block.window_size, Block_C)
                # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                ATTN_B_, ATTN_N, ATTN_C = x_windows.shape

                qkv = block.attn.qkv(x_windows) # (ATTN_B_, ATTN_N, 3 * ATTN_C)
                qkv = qkv.reshape(ATTN_B_, ATTN_N, 3, block.attn.num_heads, ATTN_C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # q: Tensor(ATTN_B_, block.attn.num_heads, ATTN_N, ATTN_C // block.attn.num_heads)
                q = q * block.attn.scale
                attn = self.matmul(q, k.transpose(-2, -1))

                relative_position_bias = block.attn.relative_position_bias_table[
                    block.attn.relative_position_index.view(-1)
                ].view(
                    block.attn.window_size[0] * block.attn.window_size[1], block.attn.window_size[0] * block.attn.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = self.add(attn, relative_position_bias.unsqueeze(0))

                if attn_mask is not None:
                    nW = attn_mask.shape[0]
                    attn = attn.view(ATTN_B_ // nW, nW, block.attn.num_heads, ATTN_N, ATTN_N) + attn_mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, block.attn.num_heads, ATTN_N, ATTN_N)
                    attn = block.attn.softmax(attn)
                else:
                    attn = block.attn.softmax(attn)

                attn = block.attn.attn_drop(attn)

                x_windows = self.matmul(attn, v).transpose(1, 2).reshape(ATTN_B_, ATTN_N, ATTN_C)
                x_windows = block.attn.proj(x_windows)
                x_windows = block.attn.proj_drop(x_windows)

                attn_windows = x_windows

                # merge windows
                attn_windows = attn_windows.view(-1, block.window_size, block.window_size, Block_C)
                shifted_x = window_reverse(attn_windows, block.window_size, Hp, Wp)  # B H' W' C

                # reverse cyclic shift
                if block.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                else:
                    x = shifted_x

                if pad_r > 0 or pad_b > 0:
                    x = x[:, :Block_H, :Block_W, :].contiguous()

                x = x.view(Block_B, Block_H * Block_W, Block_C)

                # FFN
                x = self.add(shortcut, block.drop_path(x))

                dindice_for_embeddings = torch.nonzero(dmap_layer.view(-1) == 1, as_tuple=False).view(-1)
                x_sel = x.reshape(-1, Block_C)
                x_sel = F.embedding(dindice_for_embeddings, x_sel)
                mlp_x_selected = block.drop_path(block.mlp(block.norm2(x_sel)))
                x_sel = self.add(x_sel, mlp_x_selected)
                
                fname = f"layer{lidx}_block{bidx}_ffn_out"
                if fname in anchor_features:
                    x_cached = anchor_features[fname].clone()
                    x_cached[dindice_for_embeddings] = x_sel
                    x = x_cached
                else:
                    x_cached = x.view(-1, Block_C).clone()
                    x_cached[dindice_for_embeddings] = x_sel
                    x = x_cached
                new_cache_feature[fname] = x.clone()
                x = x.unsqueeze(0)
                
            
            if layer.downsample is not None:
                x_down = layer.downsample(x, LH, LW)
                Wh, Ww = (LH + 1) // 2, (LW + 1) // 2
                x_out, H, W, x, Wh, Ww = x, LH, LW, x_down, Wh, Ww
            else:
                x_out, H, W, x, Wh, Ww = x, LH, LW, x, LH, LW

            if lidx in swin_model.out_indices:
                norm_layer = getattr(swin_model, f"norm{lidx}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, swin_model.num_features[lidx]).permute(0, 3, 1, 2).contiguous()
                outs["p{}".format(lidx)] = out

        bottom_up_features = outs

        if only_backbone:
            return ([], [], []), new_cache_feature, []

        # FPN
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        ## reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            ## Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            ## Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        
        features = {f: res for f, res in zip(backbone._out_features, results)}
        
        # Post-process features
        proposals, _ = base_model.proposal_generator(images, features, None)
        results, _ = base_model.roi_heads(images, features, proposals, None)

        predictions = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # Process predictions
        boxes = predictions[0]["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions[0]["instances"].pred_classes.cpu().numpy()
        scores = predictions[0]["instances"].scores.cpu().numpy()
        pred_masks = predictions[0]["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks

    def forward_maskvd(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cache_feature = {}

        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device).half()
        batched_inputs = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        # Preprocess image
        base_model = self.base_model
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )
        
        # Backbone
        backbone = base_model.backbone
        swin_model = backbone.bottom_up
        x = images.tensor
        
        x = swin_model.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        ## projection without APE
        x = x.flatten(2).transpose(1, 2)
        x = swin_model.pos_drop(x)

        outs = {}
        for lidx in range(swin_model.num_layers):
            # Swin Transformer Layer
            layer = swin_model.layers[lidx]
            LH, LW = Wh, Ww
            
            ## downsample the dirtiness map to the current layer's feature map size
            dmap_layer = dirtiness_map.clone().squeeze(-1).unsqueeze(0)  # 1, 1, LH, LW
            dmap_layer = F.interpolate(dmap_layer, size=(LH, LW), mode="area")
            dmap_layer = (dmap_layer > 0).float()
            dmap_layer = dmap_layer.unsqueeze(-1).squeeze(0)  # 1, LH, LW, 1

            Hp = int(np.ceil(LH / layer.window_size)) * layer.window_size
            Wp = int(np.ceil(LW / layer.window_size)) * layer.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
            h_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            w_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(
                img_mask, layer.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, layer.window_size * layer.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )

            for bidx, block in enumerate(layer.blocks):
                # Swin Transformer Block
                block.H, block.W = LH, LW
                Block_B, Block_L, Block_C = x.shape
                Block_H, Block_W = block.H, block.W

                shortcut = x
                x = block.norm1(x)
                x = x.view(Block_B, Block_H, Block_W, Block_C)

                # pad feature maps to multiples of window size
                pad_l = pad_t = 0
                pad_r = (block.window_size - Block_W % block.window_size) % block.window_size
                pad_b = (block.window_size - Block_H % block.window_size) % block.window_size
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                _, Hp, Wp, _ = x.shape

                # cyclic shift
                if block.shift_size > 0:
                    shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                    attn_mask = attn_mask
                else:
                    shifted_x = x
                    attn_mask = None

                # partition windows
                x_windows, _ = window_partition(shifted_x, block.window_size)
                dmap_windows, _ = window_partition(dmap_layer, block.window_size)
                # nW*B, window_size, window_size, C
                
                x_windows = x_windows.view(-1, block.window_size * block.window_size, Block_C)
                # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                ATTN_B_, ATTN_N, ATTN_C = x_windows.shape

                dmap_for_embeddings = dmap_windows.mean(dim=-1).mean(dim=-1).mean(dim=-1)
                dindice_for_embeddings = torch.nonzero(dmap_for_embeddings, as_tuple=False).view(-1)
                num_sel_windows = dindice_for_embeddings.shape[0]
                

                x_windows_sel = x_windows.reshape(ATTN_B_, -1)
                fname = f"layer{lidx}_block{bidx}_dindice_for_embeddings"
                if fname in anchor_features:
                    x_windows_cached = anchor_features[fname]
                    x_windows_cached[dindice_for_embeddings] = x_windows_sel[dindice_for_embeddings]
                    x_windows_sel = x_windows_cached
                new_cache_feature[fname] = x_windows_sel.clone()

                x_windows_sel = x_windows_sel.reshape(x_windows_sel.shape[0], ATTN_N, ATTN_C)
                
                qkv_sel = block.attn.qkv(x_windows_sel) # (ATTN_B_, ATTN_N, ATTN_C * 3)
                qkv_sel = qkv_sel.reshape(ATTN_B_, -1)

                qkv_sel = qkv_sel.reshape(ATTN_B_, ATTN_N, 3, block.attn.num_heads, ATTN_C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q_sel = qkv_sel[0] * block.attn.scale
                k_sel = qkv_sel[1]
                v_sel = qkv_sel[2]

                attn_sel = self.matmul(q_sel, k_sel.transpose(-2, -1))

                relative_position_bias = block.attn.relative_position_bias_table[
                    block.attn.relative_position_index.view(-1)
                ].view(
                    block.attn.window_size[0] * block.attn.window_size[1], block.attn.window_size[0] * block.attn.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = self.add(attn_sel, relative_position_bias.unsqueeze(0))

                if attn_mask is not None:
                    nW = attn_mask.shape[0]
                    attn = attn.view(ATTN_B_ // nW, nW, block.attn.num_heads, ATTN_N, ATTN_N) + attn_mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, block.attn.num_heads, ATTN_N, ATTN_N)
                    attn = block.attn.softmax(attn)
                else:
                    attn = block.attn.softmax(attn)

                attn = block.attn.attn_drop(attn)

                x_windows = self.matmul(attn, v_sel).transpose(1, 2).reshape(ATTN_B_, ATTN_N, ATTN_C)
                x_windows = block.attn.proj(x_windows)
                x_windows = block.attn.proj_drop(x_windows)

                attn_windows = x_windows

                # merge windows
                attn_windows = attn_windows.view(-1, block.window_size, block.window_size, Block_C)
                shifted_x = window_reverse(attn_windows, block.window_size, Hp, Wp)  # B H' W' C

                # reverse cyclic shift
                if block.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                else:
                    x = shifted_x

                if pad_r > 0 or pad_b > 0:
                    x = x[:, :Block_H, :Block_W, :].contiguous()

                x = x.view(Block_B, Block_H * Block_W, Block_C)

                # FFN
                x = self.add(shortcut, block.drop_path(x))

                dindice_for_embeddings = torch.nonzero(dmap_layer.view(-1) == 1, as_tuple=False).view(-1)
                x_sel = x.reshape(-1, Block_C)
                x_sel = F.embedding(dindice_for_embeddings, x_sel)
                mlp_x_selected = block.drop_path(block.mlp(block.norm2(x_sel)))
                x_sel = self.add(x_sel, mlp_x_selected)

                x.view(-1, Block_C)[dindice_for_embeddings] = x_sel

            # x: (B, H*W, C)
            fname = f"layer{lidx}_out"
            if fname in anchor_features:
                x_cached = anchor_features[fname]
                x_cached[:, dindice_for_embeddings] = x[:, dindice_for_embeddings]
                x = x_cached
            new_cache_feature[fname] = x.clone()
                
            
            if layer.downsample is not None:
                x_down = layer.downsample(x, LH, LW)
                Wh, Ww = (LH + 1) // 2, (LW + 1) // 2
                x_out, H, W, x, Wh, Ww = x, LH, LW, x_down, Wh, Ww
            else:
                x_out, H, W, x, Wh, Ww = x, LH, LW, x, LH, LW

            if lidx in swin_model.out_indices:
                norm_layer = getattr(swin_model, f"norm{lidx}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, swin_model.num_features[lidx]).permute(0, 3, 1, 2).contiguous()
                outs["p{}".format(lidx)] = out

        bottom_up_features = outs

        if only_backbone:
            return ([], [], []), new_cache_feature, []

        # FPN
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        ## reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            ## Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            ## Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        
        features = {f: res for f, res in zip(backbone._out_features, results)}
        
        # Post-process features
        proposals, _ = base_model.proposal_generator(images, features, None)
        results, _ = base_model.roi_heads(images, features, proposals, None)

        predictions = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # Process predictions
        boxes = predictions[0]["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions[0]["instances"].pred_classes.cpu().numpy()
        scores = predictions[0]["instances"].scores.cpu().numpy()
        pred_masks = predictions[0]["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks
    
    def forward_stgt(
        self, 
        image_ndarray: np.ndarray,
        anchor_features: Dict[str, torch.Tensor] = {},
        dirtiness_map: torch.Tensor = torch.ones(1, 256, 256, 1, device="cuda"),
        only_backbone: bool = False,
    ):
        new_cache_feature = {}

        # convert to tensor
        image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8).permute(2, 0, 1).to(self.device).half()
        batched_inputs = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]

        # Preprocess image
        base_model = self.base_model
        # preprocess
        images = [self.base_model._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.base_model.pixel_mean) / self.base_model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.base_model.backbone.size_divisibility,
            padding_constraints={"size_divisibility": self.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
        )
        
        # Backbone
        backbone = base_model.backbone
        swin_model = backbone.bottom_up
        x = images.tensor
        
        x = swin_model.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        ## projection without APE
        x = x.flatten(2).transpose(1, 2)
        x = swin_model.pos_drop(x)

        outs = {}
        for lidx in range(swin_model.num_layers):
            # Swin Transformer Layer
            layer = swin_model.layers[lidx]
            LH, LW = Wh, Ww
            
            ## downsample the dirtiness map to the current layer's feature map size
            dmap_layer = dirtiness_map.clone().squeeze(-1).unsqueeze(0)  # 1, 1, LH, LW
            dmap_layer = F.interpolate(dmap_layer, size=(LH, LW), mode="area")
            dmap_layer = (dmap_layer > 0).float()
            dmap_layer = dmap_layer.unsqueeze(-1).squeeze(0)  # 1, LH, LW, 1

            Hp = int(np.ceil(LH / layer.window_size)) * layer.window_size
            Wp = int(np.ceil(LW / layer.window_size)) * layer.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
            h_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            w_slices = (
                slice(0, -layer.window_size),
                slice(-layer.window_size, -layer.shift_size),
                slice(-layer.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(
                img_mask, layer.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, layer.window_size * layer.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )

            for bidx, block in enumerate(layer.blocks):
                # Swin Transformer Block
                block.H, block.W = LH, LW
                Block_B, Block_L, Block_C = x.shape
                Block_H, Block_W = block.H, block.W

                shortcut = x
                x = block.norm1(x)
                x = x.view(Block_B, Block_H, Block_W, Block_C)

                # pad feature maps to multiples of window size
                pad_l = pad_t = 0
                pad_r = (block.window_size - Block_W % block.window_size) % block.window_size
                pad_b = (block.window_size - Block_H % block.window_size) % block.window_size
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                _, Hp, Wp, _ = x.shape

                # cyclic shift
                if block.shift_size > 0:
                    shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
                    attn_mask = attn_mask
                else:
                    shifted_x = x
                    attn_mask = None

                # partition windows
                x_windows, _ = window_partition(shifted_x, block.window_size)
                dmap_windows, _ = window_partition(dmap_layer, block.window_size)
                # nW*B, window_size, window_size, C
                
                x_windows = x_windows.view(-1, block.window_size * block.window_size, Block_C)
                # nW*B, window_size*window_size, C

                # W-MSA/SW-MSA
                ATTN_B_, ATTN_N, ATTN_C = x_windows.shape

                dmap_for_embeddings = dmap_windows.mean(dim=-1).mean(dim=-1).mean(dim=-1)
                dindice_for_embeddings = torch.nonzero(dmap_for_embeddings, as_tuple=False).view(-1)
                num_sel_windows = dindice_for_embeddings.shape[0]
                
                x_windows_sel = x_windows.reshape(ATTN_B_, -1)
                x_windows_sel = F.embedding(dindice_for_embeddings, x_windows_sel)
                x_windows_sel = x_windows_sel.reshape(x_windows_sel.shape[0], ATTN_N, ATTN_C)
                
                qkv_sel = block.attn.qkv(x_windows_sel) # (ATTN_B_, ATTN_N, ATTN_C * 3)
                qkv_sel = qkv_sel.reshape(num_sel_windows, -1)

                fname = f"layer{lidx}_block{bidx}_qkv"
                if fname in anchor_features:
                    qkv_cached = anchor_features[fname]
                    qkv = qkv_cached.clone()
                    qkv[dindice_for_embeddings, :] = qkv_sel
                else:
                    qkv = torch.zeros(ATTN_B_, ATTN_N * ATTN_C * 3, device=x.device)
                    qkv[dindice_for_embeddings, :] = qkv_sel
                new_cache_feature[fname] = qkv.clone()

                qkv = qkv.reshape(ATTN_B_, ATTN_N, 3, block.attn.num_heads, ATTN_C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # q: Tensor(ATTN_B_, block.attn.num_heads, ATTN_N, ATTN_C // block.attn.num_heads)
                q = q * block.attn.scale
                q_sel = q[dindice_for_embeddings, :, :]
                k_sel = k[dindice_for_embeddings, :, :]
                v_sel = v[dindice_for_embeddings, :, :]

                attn_sel = self.matmul(q_sel, k_sel.transpose(-2, -1))

                relative_position_bias = block.attn.relative_position_bias_table[
                    block.attn.relative_position_index.view(-1)
                ].view(
                    block.attn.window_size[0] * block.attn.window_size[1], block.attn.window_size[0] * block.attn.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn_sel = self.add(attn_sel, relative_position_bias.unsqueeze(0))

                attn = torch.zeros(ATTN_B_, block.attn.num_heads, ATTN_N, ATTN_N, device=x.device)
                attn[dindice_for_embeddings, :, :] = attn_sel

                if attn_mask is not None:
                    nW = attn_mask.shape[0]
                    attn = attn.view(ATTN_B_ // nW, nW, block.attn.num_heads, ATTN_N, ATTN_N) + attn_mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, block.attn.num_heads, ATTN_N, ATTN_N)
                    attn = block.attn.softmax(attn)
                else:
                    attn = block.attn.softmax(attn)

                attn = block.attn.attn_drop(attn)

                attn_sel = attn[dindice_for_embeddings, :, :]

                x_windows_sel = self.matmul(attn_sel, v_sel).transpose(1, 2).reshape(num_sel_windows, ATTN_N, ATTN_C)
                x_windows_sel = block.attn.proj(x_windows_sel)
                x_windows_sel = block.attn.proj_drop(x_windows_sel)

                fname = f"layer{lidx}_block{bidx}_attn_proj"
                if fname in anchor_features:
                    x_windows = anchor_features[fname].clone()
                    x_windows[dindice_for_embeddings, :] = x_windows_sel
                else:
                    x_windows = torch.zeros(ATTN_B_, ATTN_N, ATTN_C, device=x.device)
                    x_windows[dindice_for_embeddings, :] = x_windows_sel
                new_cache_feature[fname] = x_windows.clone()

                attn_windows = x_windows

                # merge windows
                attn_windows = attn_windows.view(-1, block.window_size, block.window_size, Block_C)
                shifted_x = window_reverse(attn_windows, block.window_size, Hp, Wp)  # B H' W' C

                # reverse cyclic shift
                if block.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
                else:
                    x = shifted_x

                if pad_r > 0 or pad_b > 0:
                    x = x[:, :Block_H, :Block_W, :].contiguous()

                x = x.view(Block_B, Block_H * Block_W, Block_C)

                # FFN
                x = self.add(shortcut, block.drop_path(x))

                dindice_for_embeddings = torch.nonzero(dmap_layer.view(-1) == 1, as_tuple=False).view(-1)
                x_sel = x.reshape(-1, Block_C)
                x_sel = F.embedding(dindice_for_embeddings, x_sel)
                mlp_x_selected = block.drop_path(block.mlp(block.norm2(x_sel)))
                x_sel = self.add(x_sel, mlp_x_selected)
                
                fname = f"layer{lidx}_block{bidx}_ffn_out"
                if fname in anchor_features:
                    x_cached = anchor_features[fname].clone()
                    x_cached[dindice_for_embeddings] = x_sel
                    x = x_cached
                else:
                    x_cached = x.view(-1, Block_C).clone()
                    x_cached[dindice_for_embeddings] = x_sel
                    x = x_cached
                new_cache_feature[fname] = x.clone()
                x = x.unsqueeze(0)
                
            
            if layer.downsample is not None:
                x_down = layer.downsample(x, LH, LW)
                Wh, Ww = (LH + 1) // 2, (LW + 1) // 2
                x_out, H, W, x, Wh, Ww = x, LH, LW, x_down, Wh, Ww
            else:
                x_out, H, W, x, Wh, Ww = x, LH, LW, x, LH, LW

            if lidx in swin_model.out_indices:
                norm_layer = getattr(swin_model, f"norm{lidx}")
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, swin_model.num_features[lidx]).permute(0, 3, 1, 2).contiguous()
                outs["p{}".format(lidx)] = out

        bottom_up_features = outs

        if only_backbone:
            return ([], [], []), new_cache_feature, []

        # FPN
        results = []
        prev_features = backbone.lateral_convs[0](bottom_up_features[backbone.in_features[-1]])
        results.append(backbone.output_convs[0](prev_features))

        ## reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(backbone.lateral_convs, backbone.output_convs)
        ):
            ## Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            ## Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if backbone._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if backbone.top_block is not None:
            if backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[backbone._out_features.index(backbone.top_block.in_feature)]
            results.extend(backbone.top_block(top_block_in_feature))
        
        features = {f: res for f, res in zip(backbone._out_features, results)}
        
        # Post-process features
        proposals, _ = base_model.proposal_generator(images, features, None)
        results, _ = base_model.roi_heads(images, features, proposals, None)

        predictions = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # Process predictions
        boxes = predictions[0]["instances"].pred_boxes.tensor.cpu().numpy()
        labels = predictions[0]["instances"].pred_classes.cpu().numpy()
        scores = predictions[0]["instances"].scores.cpu().numpy()
        pred_masks = predictions[0]["instances"].pred_masks.cpu().numpy()

        return (boxes, labels, scores), new_cache_feature, pred_masks
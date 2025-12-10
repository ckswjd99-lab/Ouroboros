import torch
import cv2
import numpy as np

import matplotlib.pyplot as plt

from ipconv.models import MaskedRCNN_ViT_B_FPN_Contexted
from ipconv.models.ViTDet.structures import ImageList

def extract_feature(model, image_ndarray):
    image_tensor = torch.tensor(image_ndarray, dtype=torch.uint8, device=model.device).permute(2, 0, 1)
    input = [{"image": image_tensor, "height": image_tensor.shape[-2], "width": image_tensor.shape[-1]}]
    
    images = [model.base_model._move_to_current_device(x["image"]) for x in input]
    images = [(x - model.base_model.pixel_mean) / model.base_model.pixel_std for x in images]
    images = ImageList.from_tensors(
        images,
        model.base_model.backbone.size_divisibility,
        padding_constraints={"size_divisibility": model.base_model.backbone.size_divisibility, "padding_constraints": image_ndarray.shape[0]},
    )

    backbone = model.base_model.backbone
    net = backbone.net

    tokens = net.patch_embed(images.tensor).squeeze(0)

    return tokens


@torch.no_grad()
def main():
    # Save absdiff between original and shifted image
    model = MaskedRCNN_ViT_B_FPN_Contexted("cuda")
    model.load_weight('./weights/model_final_61ccd1.pkl')
    model.eval()

    image_path = "horsejump_high_00000.jpg"
    image_ndarray = cv2.imread(image_path)
    img_H, img_W, _ = image_ndarray.shape
    
    SCALING = 2/3
    image_ndarray = cv2.resize(image_ndarray, (int(img_W * SCALING), int(img_H * SCALING)), interpolation=cv2.INTER_LINEAR)

    RECT_SIZE = 16 * 12
    H_OFFSET, W_OFFSET = 50, 50
    T_CHANNEL = 1
    SHIFT_H, SHIFT_W = 0, 16
    VIS_ALPHA = 6
    VIS_BETA = -130 * VIS_ALPHA
    VIS_GAMMA = 1.6
    VIS_CMAP = "inferno"
    VIS_FONTSIZE = 20

    # token from original image
    image_ndarray = image_ndarray[H_OFFSET:H_OFFSET + RECT_SIZE, W_OFFSET:W_OFFSET + RECT_SIZE]
    
    tokens = extract_feature(model, image_ndarray)[:, :, T_CHANNEL].cpu().numpy()

    # token from shifted image
    image_ndarray_shifted = np.zeros_like(image_ndarray)
    ih_0 = max(0, SHIFT_H)
    iw_0 = max(0, SHIFT_W)
    ih_1 = min(RECT_SIZE, SHIFT_H + RECT_SIZE)
    iw_1 = min(RECT_SIZE, SHIFT_W + RECT_SIZE)
    image_ndarray_shifted[ih_0:ih_1, iw_0:iw_1] = image_ndarray[max(0, -SHIFT_H):min(RECT_SIZE, RECT_SIZE - SHIFT_H),
                                                                 max(0, -SHIFT_W):min(RECT_SIZE, RECT_SIZE - SHIFT_W)]

    tokens_shifted = extract_feature(model, image_ndarray_shifted)[:, :, T_CHANNEL].cpu().numpy()

    # token from half-shifted image
    image_ndarray_hshifted = np.zeros_like(image_ndarray)
    ih_0 = max(0, SHIFT_H // 2)
    iw_0 = max(0, SHIFT_W // 2)
    ih_1 = min(RECT_SIZE, SHIFT_H // 2 + RECT_SIZE)
    iw_1 = min(RECT_SIZE, SHIFT_W // 2 + RECT_SIZE)
    image_ndarray_hshifted[ih_0:ih_1, iw_0:iw_1] = image_ndarray[max(0, -SHIFT_H // 2):min(RECT_SIZE, RECT_SIZE - SHIFT_H // 2),
                                                                 max(0, -SHIFT_W // 2):min(RECT_SIZE, RECT_SIZE - SHIFT_W // 2)]

    tokens_hshifted = extract_feature(model, image_ndarray_hshifted)[:, :, T_CHANNEL].cpu().numpy()

    # convert to RGB for plotting
    image_ndarray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2RGB)
    image_ndarray_shifted = cv2.cvtColor(image_ndarray_shifted, cv2.COLOR_BGR2RGB)
    image_ndarray_hshifted = cv2.cvtColor(image_ndarray_hshifted, cv2.COLOR_BGR2RGB)

    absdiff_img = cv2.absdiff(image_ndarray, image_ndarray_shifted)
    absdiff_img = cv2.cvtColor(absdiff_img, cv2.COLOR_BGR2GRAY)
    plt.imsave("figs/fig_feature_img_absdiff.png", absdiff_img, cmap='gray')

    # draw grid on the image
    for i in range(0, RECT_SIZE, 16):
        cv2.line(image_ndarray, (i, 0), (i, RECT_SIZE), (255, 255, 255, 0.3), 1)
        cv2.line(image_ndarray, (0, i), (RECT_SIZE, i), (255, 255, 255, 0.3), 1)
        cv2.line(image_ndarray_shifted, (i, 0), (i, RECT_SIZE), (255, 255, 255, 0.3), 1)
        cv2.line(image_ndarray_shifted, (0, i), (RECT_SIZE, i), (255, 255, 255, 0.3), 1)
        cv2.line(image_ndarray_hshifted, (i, 0), (i, RECT_SIZE), (255, 255, 255, 0.3), 1)
        cv2.line(image_ndarray_hshifted, (0, i), (RECT_SIZE, i), (255, 255, 255, 0.3), 1)

    # normalize tokens for visualization
    gmax = max(tokens.max(), tokens_shifted.max(), tokens_hshifted.max())
    gmin = min(tokens.min(), tokens_shifted.min(), tokens_hshifted.min())
    tokens          = ((tokens - gmin) / (gmax - gmin)).astype(np.float32) ** (1/VIS_GAMMA)
    tokens_shifted  = ((tokens_shifted - gmin) / (gmax - gmin)).astype(np.float32) ** (1/VIS_GAMMA)
    tokens_hshifted = ((tokens_hshifted - gmin) / (gmax - gmin)).astype(np.float32) ** (1/VIS_GAMMA)

    tokens = (tokens * 255)
    tokens_shifted = (tokens_shifted * 255)
    tokens_hshifted = (tokens_hshifted * 255)

    tokens = cv2.convertScaleAbs(tokens, alpha=VIS_ALPHA, beta=VIS_BETA)
    tokens_shifted = cv2.convertScaleAbs(tokens_shifted, alpha=VIS_ALPHA, beta=VIS_BETA)
    tokens_hshifted = cv2.convertScaleAbs(tokens_hshifted, alpha=VIS_ALPHA, beta=VIS_BETA)

    tokens = 255 - tokens
    tokens_shifted = 255 - tokens_shifted
    tokens_hshifted = 255 - tokens_hshifted

    # plot image and tokens (subfigures)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(image_ndarray)
    ax[0, 0].set_title("Original Image", fontsize=VIS_FONTSIZE)
    ax[0, 0].axis("off")
    ax[1, 0].imshow(tokens, cmap=VIS_CMAP)
    ax[1, 0].set_title(f"Feature (Channel {T_CHANNEL})", fontsize=VIS_FONTSIZE)
    ax[1, 0].axis("off")
    ax[0, 1].imshow(image_ndarray_shifted)
    ax[0, 1].set_title("Patch-level Shift", fontsize=VIS_FONTSIZE)
    ax[0, 1].axis("off")
    ax[1, 1].imshow(tokens_shifted, cmap=VIS_CMAP)
    ax[1, 1].set_title(f"Feature (Channel {T_CHANNEL})", fontsize=VIS_FONTSIZE)
    ax[1, 1].axis("off")
    ax[0, 2].imshow(image_ndarray_hshifted)
    ax[0, 2].set_title("Sub-Patch Level Shift", fontsize=VIS_FONTSIZE)
    ax[0, 2].axis("off")
    ax[1, 2].imshow(tokens_hshifted, cmap=VIS_CMAP)
    ax[1, 2].set_title(f"Feature (Channel {T_CHANNEL})", fontsize=VIS_FONTSIZE)
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("figs/fig_feature.png")

    # Save each image and feature map separately
    plt.imsave("figs/fig_feature_img_orig.png", image_ndarray)
    plt.imsave("figs/fig_feature_img_shifted.png", image_ndarray_shifted)
    plt.imsave("figs/fig_feature_img_hshifted.png", image_ndarray_hshifted)
    plt.imsave("figs/fig_feature_token_orig.png", tokens, cmap=VIS_CMAP)
    plt.imsave("figs/fig_feature_token_shifted.png", tokens_shifted, cmap=VIS_CMAP)
    plt.imsave("figs/fig_feature_token_hshifted.png", tokens_hshifted, cmap=VIS_CMAP)


if __name__ == "__main__":
    main()
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----- 설정 -----
ROOT = Path("/data/DAVIS2017_trainval/Annotations/480p")
W, H = 854, 480  # (width, height)

# ----- 전역 히트맵 누적 -----
heatmap = np.zeros((H, W), dtype=np.uint32)

seq_dirs = [p for p in ROOT.iterdir() if p.is_dir()]
seq_dirs.sort()

total_imgs = 0
for seq in seq_dirs:
    pngs = sorted(seq.glob("*.png"))
    for png in pngs:
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[warn] fail to read: {png}")
            continue
        if img.shape[::-1] != (W, H):  # (width,height) 비교
            # raise ValueError(f"Unexpected size {img.shape} at {png}, expected (480,854)")
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

        # non-zero 픽셀을 1로, 나머지 0으로 만든 뒤 누적
        heatmap += (img > 0).astype(np.uint32)
        total_imgs += 1

print(f"Scanned {len(seq_dirs)} sequences, {total_imgs} images.")
np.save("heatmap_counts.npy", heatmap)

# ----- 시각화 (정규화) -----
if heatmap.max() > 0:
    viz = heatmap.astype(np.float32) / heatmap.max()
else:
    viz = heatmap.astype(np.float32)

plt.figure(figsize=(10, 6))
plt.imshow(viz, cmap='hot', interpolation='nearest')
plt.colorbar(label='Normalized Count')
plt.axis('off')
plt.title("DAVIS2017 TrainVal Annotations Heatmap (non-zero counts, normalized)")
plt.savefig("heatmap.png", dpi=200)
plt.close()

print("Saved: heatmap_counts.npy, heatmap.png")

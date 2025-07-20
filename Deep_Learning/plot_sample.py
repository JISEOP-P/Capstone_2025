import numpy as np
import matplotlib.pyplot as plt
import os

# ========== CONFIG ==========
BASE_NAME = "GDN_1_s4_v3_023_G"
PREPROCESS_DIR = "dataset/preprocessed_02"
SAVE_DIR = "Test_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Load Data ==========
rtm_path = os.path.join(PREPROCESS_DIR, f"{BASE_NAME}_rtm.npy")
dtm_path = os.path.join(PREPROCESS_DIR, f"{BASE_NAME}_dtm.npy")

rtm = np.load(rtm_path)  # (224,224,3)
dtm = np.load(dtm_path)  # (224,224,3)

# ========== Plot ==========
fig, axes = plt.subplots(2, 3, figsize=(12,8), constrained_layout=True)
fig.suptitle(f"{BASE_NAME} RTM / DTM Visualization", fontsize=16)

vmax = max(np.max(rtm), np.max(dtm))
im = None

# RTM
for i in range(3):
    ax = axes[0,i]
    im = ax.imshow(rtm[:,:,i], cmap='jet', vmin=0, vmax=vmax, aspect='auto')
    ax.set_title(f"RTM Rx{i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Range")

# DTM
for i in range(3):
    ax = axes[1,i]
    im = ax.imshow(dtm[:,:,i], cmap='jet', vmin=0, vmax=vmax, aspect='auto')
    ax.set_title(f"DTM Rx{i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Doppler")

# 공통 컬러바
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label("Normalized Magnitude")

# 저장
save_path = os.path.join(SAVE_DIR, f"{BASE_NAME}_plot.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"[Saved] {save_path}")

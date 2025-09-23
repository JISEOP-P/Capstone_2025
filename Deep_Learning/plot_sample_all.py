import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# ========== CONFIG ==========
RAW_SAMPLE_DIR = "dataset/raw"
PREPROCESS_DIR = "dataset/preprocessed"
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Helper Function ==========
def visualize_pair(base_name, rtm_path, dtm_path, save_dir):
    rtm = np.load(rtm_path)
    dtm = np.load(dtm_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(12,8), constrained_layout=True)
    fig.suptitle(f"{base_name} RTM / DTM Visualization", fontsize=16)

    vmax = max(np.max(rtm), np.max(dtm))
    im = None

    for i in range(3):
        ax = axes[0,i]
        im = ax.imshow(rtm[:,:,i], cmap='jet', vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(f"RTM Rx{i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Range")

    for i in range(3):
        ax = axes[1,i]
        im = ax.imshow(dtm[:,:,i], cmap='jet', vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(f"DTM Rx{i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Doppler")

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Normalized Magnitude")

    save_path = os.path.join(save_dir, f"{base_name}_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# ========== Main Loop ==========
raw_files = glob(os.path.join(RAW_SAMPLE_DIR, "*.npy"))

for raw_file in raw_files:
    base_name = os.path.basename(raw_file).replace(".npy", "")
    rtm_path = os.path.join(PREPROCESS_DIR, f"{base_name}_rtm.npy")
    dtm_path = os.path.join(PREPROCESS_DIR, f"{base_name}_dtm.npy")
    
    if os.path.exists(rtm_path) and os.path.exists(dtm_path):
        visualize_pair(base_name, rtm_path, dtm_path, SAVE_DIR)
    else:
        print(f"[Skip] {base_name}: RTM/DTM not found in preprocess dir")

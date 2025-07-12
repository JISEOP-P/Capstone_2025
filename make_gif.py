import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# ===== CONFIG =====
DATA_PATH = "dataset/raw/GDN_0_s1_v1_001_D.npy"  # 원본 샘플 파일
SAVE_DIR = "gif"
SAVE_PATH = os.path.join(SAVE_DIR, "0.gif")
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== LOAD DATA =====
frames = np.load(DATA_PATH)  # shape: (50, 3, 256, 128)
num_samples = 128
num_chirps = 256

# doppler bins for plotting
doppler_bins = np.fft.fftshift(np.fft.fftfreq(num_chirps)) * num_chirps
doppler_bins = doppler_bins.astype(int)

image_frames = []

# ===== PROCESS EACH FRAME =====
for idx, frame in enumerate(frames):  # (3,256,128)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Frame {idx}", fontsize=16)

    for i in range(3):
        data = frame[i]  # (256,128)
        # Range FFT
        range_fft = np.fft.fft(data, axis=1)
        # Doppler FFT
        doppler_fft = np.fft.fft(range_fft, axis=0)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
        magnitude_log = np.log1p(np.abs(doppler_fft))

        # Plot
        im = axs[i].imshow(
            magnitude_log,
            aspect='auto',
            cmap='jet',
            origin='lower',
            extent=[0, num_samples-1, -num_chirps//2, num_chirps//2 - 1]
        )
        axs[i].set_title(f"RX{i}")
        axs[i].set_xlabel(f"Sample index (0~{num_samples-1})")
        axs[i].set_ylabel(f"Doppler FFT bin (-{num_chirps//2}~{num_chirps//2 - 1})")
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))

        plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    image_frames.append(img.convert("P"))

# ===== SAVE GIF =====
image_frames[0].save(
    SAVE_PATH,
    save_all=True,
    append_images=image_frames[1:],
    duration=100,
    loop=0
)

print(f"GIF saved: {SAVE_PATH}")

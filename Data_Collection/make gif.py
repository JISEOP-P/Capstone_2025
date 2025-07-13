import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# 데이터 로드
frames = np.load("Data_Collection/GDN_6_s1_v1_003_D.npy")  # shape: (50, 3, 256, 128)
image_frames = []

# sensor config
num_samples = 128
num_chirps = 256

# doppler fft 후 fftshift 범위
doppler_bins = np.fft.fftshift(np.fft.fftfreq(num_chirps)) * num_chirps
doppler_bins = doppler_bins.astype(int)  # 보통 -128 ~ +127 로 나옴

for idx, frame in enumerate(frames):  # frame: (3, 256, 128)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Frame {idx}", fontsize=16)

    for i in range(3):
        data = frame[i]  # shape: (256 chirps, 128 samples)

        # 1. Range FFT
        range_fft = np.fft.fft(data, axis=1)
        # 2. Doppler FFT
        doppler_fft = np.fft.fft(range_fft, axis=0)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
        # 3. Magnitude & log1p
        magnitude_log = np.log1p(np.abs(doppler_fft))

        # 시각화
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

        # tick을 적당히 줄이기
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(5))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(5))

        plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 버퍼에 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    image_frames.append(img.convert("P"))

# GIF 저장
image_frames[0].save(
    "radar_sequence.gif",
    save_all=True,
    append_images=image_frames[1:],
    duration=100,
    loop=0
)

print("GIF saved: radar_sequence.gif")

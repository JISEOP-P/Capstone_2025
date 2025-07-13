import numpy as np
import os
import csv
import warnings
import multiprocessing as mp
import cv2
from scipy.signal import windows
from tqdm import tqdm

# === Config ===
RAW_DIR = "dataset/raw"
SAVE_DIR = "dataset/preprocessed"
LABEL_CSV = os.path.join(SAVE_DIR, "labels.csv")
NUM_WORKERS = mp.cpu_count()
os.makedirs(SAVE_DIR, exist_ok=True)


# === Utils ===
def min_max_normalize(x: np.ndarray) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min == 0:
        return np.zeros_like(x)
    normed = (x - x_min) / (x_max - x_min)
    return np.nan_to_num(normed)

def save_label_csv(csv_path: str, base_filename: str, label: int) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['file', 'label'])
        writer.writerow([base_filename, label])

def adaptive_threshold_normalize(img):
    mu = np.mean(img)
    sigma = np.std(img)
    thr = mu + sigma
    img[img < thr] = 0
    img = img - thr
    img[img < 0] = 0
    if np.max(img) != 0:
        img = img / np.max(img)
    return img

def fast_cfar_2d(image, guard_cells=2, reference_cells=10, scale=1.5):
    M, N = image.shape

    total_cells = (reference_cells * 2 + guard_cells * 2 + 1) ** 2
    guard_cells_count = (guard_cells * 2 + 1) ** 2

    num_reference_cells = total_cells - guard_cells_count

    # make kernel (1s everywhere except guard + CUT area)
    kernel_size = reference_cells * 2 + guard_cells * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # zero out guard + CUT region
    start = reference_cells
    end = reference_cells + 2 * guard_cells + 1
    kernel[start:end, start:end] = 0

    # compute local sum using convolution
    local_sum = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    noise_level = local_sum / num_reference_cells

    threshold_map = noise_level * scale

    # apply threshold
    cfar_mask = (image > threshold_map).astype(np.float32)

    return cfar_mask

# === MTI ===
def frequency_weighted_mti(range_fft_frames, weight_ratio=0.5):
    avg_spectrum = np.mean(np.abs(range_fft_frames), axis=0)
    threshold = np.mean(avg_spectrum)
    weights = np.where(avg_spectrum >= threshold, 1.0, weight_ratio)
    mean_fft = np.mean(range_fft_frames, axis=0)
    filtered_frames = []
    for frame in range_fft_frames:
        filtered = frame - weights * mean_fft
        filtered_frames.append(filtered)
    return np.stack(filtered_frames, axis=0)

def temporal_mti(stack):
    filtered = np.zeros_like(stack)
    for t in range(1, stack.shape[0]):
        filtered[t] = stack[t] - stack[t-1]
    return filtered[1:]

# === RTM/DTM ===
def compute_rtm_dtm(frames, rx_index, top_k=15):
    raw_stack = frames[:, :, :, rx_index]  # (50, 256, 128)
    T = raw_stack.shape[0]
    rdi_list = []

    for t in range(T):
        raw = raw_stack[t]
        if t > 0:
            raw = raw_stack[t] - raw_stack[t-1]
        raw *= windows.hamming(raw.shape[1])[None, :]
        range_fft = np.fft.fft(raw, axis=1, n=448)[:, :224]
        range_fft = frequency_weighted_mti(range_fft, weight_ratio=0.5)
        range_fft *= windows.hamming(range_fft.shape[0])[:, None]
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
        rdi = np.abs(doppler_fft)[128-112:128+112, :]
        rdi = cv2.resize(rdi, (224, 224), interpolation=cv2.INTER_LINEAR)
        rdi = adaptive_threshold_normalize(rdi)

        cfar_mask = fast_cfar_2d(rdi, guard_cells=2, reference_cells=10, scale=1.5)
        rdi = rdi * cfar_mask

        rdi_list.append(rdi)

    rdi_stack = np.stack(rdi_list, axis=0)  # (T,224,224)
    rdi_stack = temporal_mti(rdi_stack)  # temporal filtering

    rtm_list, dtm_list = [], []
    for rdi in rdi_stack:
        top_doppler_idx = np.argsort(np.sum(rdi, axis=1))[-top_k:]
        rtm_vec = np.mean(rdi[top_doppler_idx, :], axis=0)

        top_range_idx = np.argsort(np.sum(rdi, axis=0))[-top_k:]
        dtm_vec = np.mean(rdi[:, top_range_idx], axis=1)

        rtm_list.append(rtm_vec)
        dtm_list.append(dtm_vec)

    rtm = np.stack(rtm_list, axis=1)  # (224, T-2)
    dtm = np.stack(dtm_list, axis=1)  # (224, T-2)

    rtm_resized = cv2.resize(rtm, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dtm_resized = cv2.resize(dtm, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # rtm_resized = min_max_normalize(rtm_resized)
    # dtm_resized = min_max_normalize(dtm_resized)

    return rtm_resized, dtm_resized

# === Main Processor ===
def process_single_sample(file_path: str) -> None:
    try:
        frames = np.load(file_path)  # (50, 3, 256, 128)
        frames = np.transpose(frames, (0,2,3,1))  # (50, 256,128,3)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        label = int(base_filename.split("_")[1])  # 예시: sample_2.npy -> 라벨=2

        rtm_stack, dtm_stack = [], []
        for rx_idx in range(3):
            rtm, dtm = compute_rtm_dtm(frames, rx_idx)
            rtm = rtm[..., np.newaxis]
            dtm = dtm[..., np.newaxis]
            rtm_stack.append(rtm)
            dtm_stack.append(dtm)

        rtm_all = np.concatenate(rtm_stack, axis=-1)  # (224,224,3)
        dtm_all = np.concatenate(dtm_stack, axis=-1)  # (224,224,3)

        np.save(os.path.join(SAVE_DIR, f"{base_filename}_rtm.npy"), rtm_all)
        np.save(os.path.join(SAVE_DIR, f"{base_filename}_dtm.npy"), dtm_all)

        save_label_csv(LABEL_CSV, f"{base_filename}_rtm", label)
        save_label_csv(LABEL_CSV, f"{base_filename}_dtm", label)

        print(f"[Processed] {base_filename} -> RTM/DTM 저장 (label: {label})")

    except Exception as e:
        print(f"[Error] {file_path}: {e}")

def unpack_and_process(file_path):
    return process_single_sample(file_path)

def process_all_files():
    files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith('.npy')]
    with mp.Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(unpack_and_process, files), total=len(files), desc="Processing"):
            pass

if __name__ == "__main__":
    warnings.simplefilter("ignore", category=RuntimeWarning)
    process_all_files()
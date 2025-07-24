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

def fast_cfar_2d(image, guard_cells=2, reference_cells=15, scale=1.5):
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

# === RTM/DTM ===
def compute_rtm_dtm(frames, rx_index, top_k=10):
    raw_stack = frames[:, :, :, rx_index]  # (50, 256, 128)
    T = raw_stack.shape[0]
    rdi_stack = []
    rtm_list, dtm_list = [], []
    doppler_window = None


    for t in range(T):
        raw_frame = raw_stack[t]

        # === Mean subtraction (2D)
        col_mean = np.mean(raw_frame, axis=0, keepdims=True)
        mean_sub_frame = raw_frame - col_mean
        row_mean = np.mean(mean_sub_frame, axis=1, keepdims=True)
        mean_sub_frame = mean_sub_frame - row_mean

        # === Range Hamming + FFT
        X = mean_sub_frame * windows.hamming(mean_sub_frame.shape[1])[None, :]
        range_fft = np.fft.fft(X, axis=1, n=448)[:, :224]

        # === Doppler Hamming 준비
        if doppler_window is None:
            doppler_window = windows.hamming(range_fft.shape[0])[:, None]
        doppler_input = range_fft * doppler_window

        # === Doppler FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(doppler_input, axis=0), axes=0)

        # === magnitude + CFAR + log1p
        doppler_mag = np.abs(doppler_fft)
        cfar_mask = fast_cfar_2d(doppler_mag)
        doppler_mag = doppler_mag * cfar_mask
        doppler_mag = np.log1p(doppler_mag)

        rdi_stack.append(doppler_mag)

    # === top-k sum 으로 rtm, dtm 만들기
    for rdi in rdi_stack:
        rtm_vec = np.sum(np.sort(rdi, axis=0)[-top_k:, :], axis=0)
        dtm_vec = np.sum(np.sort(rdi, axis=1)[:, -top_k:], axis=1)
        rtm_list.append(rtm_vec)
        dtm_list.append(dtm_vec)

    # stack & resize
    rtm = np.stack(rtm_list, axis=1)
    dtm = np.stack(dtm_list, axis=1)

    rtm_resized = cv2.resize(rtm, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dtm_resized = cv2.resize(dtm, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    return rtm_resized, dtm_resized


# === Main Processor ===
def process_single_sample(file_path: str) -> None:
    try:
        frames = np.load(file_path)  # (50, 3, 256, 128)
        frames = np.transpose(frames, (0,2,3,1))  # (50, 256,128,3)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        label = int(base_filename.split("_")[1])  # 예시: sample_2.npy -> 라벨=2

        rtm_list, dtm_list = [], []

        rtm_stack, dtm_stack = [], []
        for rx_idx in range(3):
            rtm, dtm = compute_rtm_dtm(frames, rx_idx)
            rtm_list.append(min_max_normalize(rtm))
            dtm_list.append(min_max_normalize(dtm))

        # Stack along last axis -> (224, 224, 3)
        rtm_stack = np.stack(rtm_list, axis=-1)
        dtm_stack = np.stack(dtm_list, axis=-1)

        np.save(os.path.join(SAVE_DIR, f"{base_filename}_rtm.npy"), rtm_stack)
        np.save(os.path.join(SAVE_DIR, f"{base_filename}_dtm.npy"), dtm_stack)

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
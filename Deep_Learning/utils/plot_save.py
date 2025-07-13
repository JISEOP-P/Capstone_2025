import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def smooth_curve(x):
    """Kaiser 윈도우 기반 smoothing 함수"""
    window_len = 4
    if len(x) < window_len:
        return x
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w / w.sum(), s, mode='valid')
    half = window_len // 2
    return y[half:len(y)-half]

def safe_smooth(x):
    """데이터 길이에 따라 smoothing 여부 자동 결정"""
    return smooth_curve(x) if len(x) >= 2 else x
 
def plot_training_curve(history, model_dir, smooth=True):
    """학습 곡선 시각화 및 저장 (스무딩 + 마커 + grid 포함)"""
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    def maybe_smooth(x): return safe_smooth(x) if smooth else x

    # Accuracy
    axs[0].plot(maybe_smooth(history.history['accuracy']), label='Train Acc',
                marker='.', markevery=3)
    axs[0].plot(maybe_smooth(history.history['val_accuracy']), label='Val Acc',
                marker='.', markevery=3)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)

    # Loss
    axs[1].plot(maybe_smooth(history.history['loss']), label='Train Loss',
                marker='v', markevery=3, markersize=4)
    axs[1].plot(maybe_smooth(history.history['val_loss']), label='Val Loss',
                marker='v', markevery=3, markersize=4)
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True, alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(model_dir, f'training_curve.png')
    plt.savefig(save_path)
    plt.show()



def plot_confusion_matrix(matrix, save_path, acc=None, loss=None, labels=None):
    matrix = np.array(matrix, dtype=np.float32)

    # 행 기준 정규화
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = matrix / np.where(row_sums == 0, 1, row_sums)  # 0으로 나누기 방지

    # 넉넉한 크기로 시작
    fig, ax = plt.subplots(figsize=(8, 7))  # 가로 조금 넓히기

    sns.heatmap(norm_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)

    # 정사각형 셀 비율
    ax.set_aspect('equal')

    # X축 글자 각도와 정렬 보정
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_yticklabels(labels, rotation=0)

    # 제목 설정
    title = "Normalized Confusion Matrix (Evaluation Set)"
    if acc is not None and loss is not None:
        title += f"\nAccuracy: {acc * 100:.2f}%, Loss: {loss:.4f}"
    ax.set_title(title)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # 간격 보정
    plt.subplots_adjust(bottom=0.2)  # 글자 겹침 방지
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.show()



def save_metrics(acc, loss, save_path, param_count, y_true=None, y_pred=None, class_names=None):
    # 단일 실험 결과 저장
    df = pd.DataFrame({
        "val_accuracy": [acc],
        "val_loss": [loss]
    })
    df.to_csv(os.path.join(save_path, "metrics.csv"), index=False)

    with open(os.path.join(save_path, "params.txt"), "w") as f:
        f.write(f"Total Parameters: {param_count:,}\n")

    # 정밀도 리포트 저장
    if y_true is not None and y_pred is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        with open(os.path.join(save_path, "classification_report.txt"), "w") as f:
            f.write(report)
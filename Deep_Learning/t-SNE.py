import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# === 설정 ===
npy_dir = "dataset/preprocessed"  # 변경 가능
label_csv = os.path.join(npy_dir, "labels.csv")
save_path = os.path.join("tsne.png")

# === 데이터 로딩 ===
def load_rtm_dtm_combined(npy_dir, label_csv):
    df = pd.read_csv(label_csv)
    rtm_df = df[df['file'].str.endswith('_rtm')].copy()
    dtm_df = df[df['file'].str.endswith('_dtm')].copy()

    rtm_df['base'] = rtm_df['file'].str.replace('_rtm', '', regex=False)
    dtm_df['base'] = dtm_df['file'].str.replace('_dtm', '', regex=False)

    merged = pd.merge(rtm_df, dtm_df, on='base', suffixes=('_rtm', '_dtm'))

    data_rtm, data_dtm, data_comb, labels = [], [], [], []

    for _, row in merged.iterrows():
        rtm_path = os.path.join(npy_dir, row['file_rtm'] + ".npy")
        dtm_path = os.path.join(npy_dir, row['file_dtm'] + ".npy")
        if not os.path.exists(rtm_path) or not os.path.exists(dtm_path):
            continue
        rtm = np.load(rtm_path).flatten()
        dtm = np.load(dtm_path).flatten()
        data_rtm.append(rtm)
        data_dtm.append(dtm)
        data_comb.append(np.concatenate([rtm, dtm]))
        labels.append(row['label_rtm'])

    return np.stack(data_rtm), np.stack(data_dtm), np.stack(data_comb), np.array(labels)

# === 차원 축소 ===
def reduce_tsne(X):
    X_pca = PCA(n_components=50).fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)
    return X_tsne

# === 시각화 ===
def plot_tsne_subplot(Xs, y, titles, save_path):
    plt.figure(figsize=(18, 5))
    for i, (X, title) in enumerate(zip(Xs, titles), 1):
        plt.subplot(1, 3, i)
        for label in np.unique(y):
            idx = y == label
            plt.scatter(X[idx, 0], X[idx, 1], label=f'Class {label}', s=5, alpha=0.6)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] t-SNE plot saved to {save_path}")

# === 실행 ===
X_rtm, X_dtm, X_comb, y = load_rtm_dtm_combined(npy_dir, label_csv)
X_rtm_2d = reduce_tsne(X_rtm)
X_dtm_2d = reduce_tsne(X_dtm)
X_comb_2d = reduce_tsne(X_comb)

plot_tsne_subplot(
    [X_rtm_2d, X_dtm_2d, X_comb_2d],
    y,
    ["RTM t-SNE", "DTM t-SNE", "Combined t-SNE"],
    save_path
)

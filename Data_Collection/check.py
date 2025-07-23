import os
from collections import defaultdict, Counter

# 경로 설정
DATA_DIR = os.path.join("..", "Data_Collection", "Data_Collection_D")

# 라벨 기준(X값)으로 피실험자 카운트 저장용
label_participant_count = defaultdict(lambda: Counter())
total_label_count = Counter()

# 순회
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".npy"):
        continue

    parts = fname.split("_")
    if len(parts) < 6:
        continue  # 예외 처리

    label = parts[1]  # '10', '11', ..., '14'
    participant = parts[5].split(".")[0]  # 'G', 'D', or 'N'

    total_label_count[label] += 1
    label_participant_count[label][participant] += 1

# 출력
print("라벨(X값)별 총 샘플 수 및 참여자별 분포:\n")
for label in sorted(total_label_count.keys(), key=int):
    total = total_label_count[label]
    g = label_participant_count[label].get("G", 0)
    d = label_participant_count[label].get("D", 0)
    n = label_participant_count[label].get("N", 0)
    print(f"라벨 {label}: 총 {total}개 (G: {g}, D: {d}, N: {n})")
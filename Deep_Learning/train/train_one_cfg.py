# train_one_cfg.py  (REPLACE)
import argparse, json, time, os, pickle, gc, sys, traceback, random
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import mixed_precision
import numpy as np
SEED = 30
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# <GPU Memory>
# GPU 설정: 메모리 동적 할당
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU 메모리 할당이 동적으로 설정되었습니다.")
    except RuntimeError as e:
        print(f"GPU 메모리 설정 중 오류 발생: {e}")

# 최소 로그
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# 혼합정밀(원하면 주석)
mixed_precision.set_global_policy('mixed_float16')
# XLA는 환경 따라 메모리↑ 가능
tf.config.optimizer.set_jit(False)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_json", type=str, required=True)
parser.add_argument("--train_ids", type=str, required=True)
parser.add_argument("--val_ids", type=str, required=True)
parser.add_argument("--label_map", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
args = parser.parse_args()

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
result_path = os.path.join(out_dir, "result.json")

def dump_result(status, **kwargs):
    """결과를 파일에 저장 (stdout 믿지 않음)"""
    payload = {"status": status, **kwargs}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

try:
    # 데이터 로드
    with open(args.train_ids, "rb") as f: train_base = pickle.load(f)
    with open(args.val_ids, "rb") as f:   val_base   = pickle.load(f)
    with open(args.label_map, "rb") as f: label_map  = pickle.load(f)

    # 하이퍼
    DATASET_DIR = "dataset/preprocessed"
    BATCH_SIZE = 16
    EPOCHS = 5
    N_CLASSES = 15
    LR = 1e-3
    WD = 1e-4

    # 데이터셋
    from utils.data_loader import RadarDataset
    train_ds = RadarDataset(train_base, label_map, batch_size=BATCH_SIZE, data_dir=DATASET_DIR).get_dataset().prefetch(tf.data.AUTOTUNE)
    val_ds   = RadarDataset(val_base,   label_map, batch_size=BATCH_SIZE, data_dir=DATASET_DIR).get_dataset().prefetch(tf.data.AUTOTUNE)

    # 모델
    from model.spectranet import build_spectranet
    cfg = json.loads(args.cfg_json)
    model = build_spectranet(input_shape=(224,224,3), cfg=cfg, num_classes=N_CLASSES)
    params = model.count_params()

    # 콜백: BestSaver (필요하면 저장 경량화 가능)
    class BestSaver(tf.keras.callbacks.Callback):
        def __init__(self, out_dir):
            super().__init__()
            self.out_dir = out_dir
            os.makedirs(self.out_dir, exist_ok=True)
            self.best_acc = -np.inf
            self.best_loss = np.inf
        def on_epoch_end(self, epoch, logs=None):
            acc = logs.get('val_accuracy')
            loss = logs.get('val_loss')
            improved = False
            if acc is not None and loss is not None:
                if acc > self.best_acc:
                    improved = True
                elif np.isclose(acc, self.best_acc) and (loss < self.best_loss):
                    improved = True
            if improved:
                self.best_acc = acc
                self.best_loss = loss
                path = os.path.join(self.out_dir, "best_model.keras")
                self.model.save(path)
                # 저장 시에만 한 줄 출력 (stderr로)
                print(f"[BestSaver] epoch {epoch+1} acc={acc:.4f} loss={loss:.4f} -> {path}", file=sys.stderr, flush=True)

    opt = AdamW(learning_rate=LR, weight_decay=WD)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    _ = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0, callbacks=[BestSaver(out_dir)])
    train_sec = time.time() - t0

    # best 평가
    best_path = os.path.join(out_dir, "best_model.keras")
    if os.path.exists(best_path):
        best_model = tf.keras.models.load_model(best_path)
    else:
        best_model = model
    loss, acc = best_model.evaluate(val_ds, verbose=0)

    # 파일에 결과 저장
    dump_result(
        "ok",
        val_acc=float(acc),
        val_loss=float(loss),
        params=int(params),
        train_sec=float(train_sec)
    )

except Exception as e:
    err_tb = traceback.format_exc()
    # 에러는 stderr로
    print("[train_one_cfg.py ERROR]\n" + err_tb, file=sys.stderr, flush=True)
    dump_result("error", message=str(e))
finally:
    try:
        del model
        del best_model
    except Exception:
        pass
    tf.keras.backend.clear_session()
    gc.collect()

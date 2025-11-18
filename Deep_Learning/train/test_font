import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
import matplotlib.pyplot as plt
import pickle
import importlib





# === Dataset Loader ===
class AugmentedRadarDataset:
    def __init__(self, base_file_list, label_map, batch_size=32, shuffle=True,
                data_dir='dataset/preprocessed', augment_fn=None):
        self.base_file_list = base_file_list
        self.label_map = label_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.augment_fn = augment_fn
        self.dataset = self._build_tf_dataset()

    def _load_paths_and_labels(self):
        rtm_paths, dtm_paths, labels = [], [], []
        for base in self.base_file_list:
            rtm_path = os.path.join(self.data_dir, f"{base}_rtm.npy")
            dtm_path = os.path.join(self.data_dir, f"{base}_dtm.npy")
            if os.path.exists(rtm_path) and os.path.exists(dtm_path):
                rtm_paths.append(rtm_path.encode())
                dtm_paths.append(dtm_path.encode())
                labels.append(self.label_map[base])
        return np.array(rtm_paths), np.array(dtm_paths), np.array(labels)

    def _build_tf_dataset(self):
        rtm_paths, dtm_paths, labels = self._load_paths_and_labels()
        dataset = tf.data.Dataset.from_tensor_slices((rtm_paths, dtm_paths, labels))

        def _load_npy(rtm_path, dtm_path, label):
            rtm = tf.numpy_function(np.load, [rtm_path], tf.float32)
            dtm = tf.numpy_function(np.load, [dtm_path], tf.float32)
            if self.augment_fn is not None:
                rtm, dtm = self.augment_fn(rtm, dtm)
            rtm.set_shape([224, 224, 3])
            dtm.set_shape([224, 224, 3])
            return {"rtm_input": rtm, "dtm_input": dtm}, label

        dataset = dataset.map(_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self):
        return self.dataset



# === Custom Checkpoint ===
class CustomBestModelSaver(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.best_acc = -np.inf
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_loss = val_loss
            self.model.save(self.save_path)
            print(f"[CUSTOM CHECKPOINT] Epoch {epoch+1}: 새로운 최고 정확도 
                 {val_acc:.4f} - 모델 저장!")
        elif val_acc == self.best_acc and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save(self.save_path)
            print(f"[CUSTOM CHECKPOINT] Epoch {epoch+1}: 동률 정확도에서 
                 낮은 손실 {val_loss:.4f} - 모델 저장!")




# === LR Scheduler ===
def lr_schedule(epoch, lr):
    if epoch == 35:
        return lr * 0.1
    return lr


# === Training Function ===
def train_model(model, train_set, val_set, output_dir, 
                class_weights=None, epochs=100, learning_rate=1e-3):
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.keras")
    model.compile(optimizer=tf.keras.optimizers.AdamW(
                            learning_rate=learning_rate, weight_decay=1e-4),
                            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [CustomBestModelSaver(save_path=best_model_path), lr_scheduler]
    history = model.fit(train_set, validation_data=val_set, epochs=epochs,
                       callbacks=callbacks, verbose=2)
    return history


# === Experiment Config ===
experiment_root = "experiments/Step_03"
DATASET_DIR = "dataset/preprocessed"
os.makedirs(experiment_root, exist_ok=True)
label_csv_path = os.path.join(DATASET_DIR, "labels.csv")
split_file_path = os.path.join("splits.pkl")

n_classes = 15
batch_size = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
apply_class_weights = False
model_list = ["spectranet"]

input_shapes = {
    'rtm_input': (224,224,3),
    'dtm_input': (224,224,3)
}

class_names = [
    "hello","nice to meet you","thank you","i respect you","sign language","i love you",
    "take care","i'm sorry","be happy","welcome","enjoy your meal","aha, i see",
    "i understand","goodbye","no gesture"
]

# === Load Labels & Split ===
df = pd.read_csv(label_csv_path)
df['base_id'] = df['file'].apply(lambda x: x.rsplit('_', 1)[0])
base_df = df.drop_duplicates('base_id')
base_df['subject'] = base_df['base_id'].apply(lambda x: x.split('_')[-1])
base_df['label_subject'] = base_df['label'].astype(str) + "_" + base_df['subject']
label_subject_map = dict(zip(base_df['base_id'], base_df['label_subject']))
label_map = dict(zip(base_df['base_id'], base_df['label']))


with open(split_file_path, 'rb') as f:
    split_dict = pickle.load(f)
train_base, val_base = split_dict['train'], split_dict['val']
print("[Info] splits.pkl 로드 완료")

# === Augmentation ===
def augment_rtm_dtm(rtm, dtm, max_shift=30, stretch_range=(0.7, 1.3),
                     time_stretch_range=(0.7, 1.3)):
    rtm.set_shape([224, 224, 3])
    dtm.set_shape([224, 224, 3])
    shift = tf.random.uniform([], -max_shift, max_shift+1, dtype=tf.int32)

    def rtm_shift_fn(img):
        return tf.cond(
            shift > 0,
            lambda: tf.concat([tf.zeros([shift, 224, 3]), img[:-shift]], axis=0),
            lambda: tf.concat([img[-shift:], tf.zeros([-shift, 224, 3])], axis=0)
        )

    rtm = rtm_shift_fn(rtm)
    scale_y = tf.random.uniform([], stretch_range[0], stretch_range[1])
    mid = tf.shape(dtm)[0] // 2
    top = dtm[:mid]
    bottom = dtm[mid:]
    new_top = tf.image.resize(top, (tf.cast(tf.cast(mid, tf.float32)*scale_y, tf.int32), 224))
    new_bottom = tf.image.resize(bottom, 
                                 (tf.cast(tf.cast(224-mid, tf.float32)*scale_y, tf.int32), 224))
    dtm = tf.concat([new_top, new_bottom], 0)
    dtm = tf.image.resize_with_crop_or_pad(dtm, 224, 224)
    scale_x = tf.random.uniform([], time_stretch_range[0], time_stretch_range[1])
    new_width = tf.cast(224.0 * scale_x, tf.int32)
    rtm = tf.image.resize(rtm, (224, new_width))
    dtm = tf.image.resize(dtm, (224, new_width))
    rtm = tf.image.resize_with_crop_or_pad(rtm, 224, 224)
    dtm = tf.image.resize_with_crop_or_pad(dtm, 224, 224)
    return rtm, dtm

# === Dataset Build ===
train_data = AugmentedRadarDataset(train_base, label_map, batch_size=batch_size,
                                   data_dir=DATASET_DIR, augment_fn=augment_rtm_dtm)
val_data = AugmentedRadarDataset(val_base, label_map, batch_size=batch_size,
                                   data_dir=DATASET_DIR)
train_dataset = train_data.get_dataset()
val_dataset = val_data.get_dataset()


# === Training Loop ===
for model_name in model_list:
    print(f"\n[실험 시작] {model_name}")
    model_dir = os.path.join(experiment_root, model_name)
    os.makedirs(model_dir, exist_ok=True)

    module = importlib.import_module(f"model.{model_name}")
    build_fn = getattr(module, f"build_{model_name}")
    model = build_fn(input_shapes=input_shapes, num_classes=n_classes)

    plot_model(model, to_file=os.path.join(model_dir, "architecture.png"),
               show_shapes=True, expand_nested=False)

    history = train_model(model, train_dataset, val_dataset, output_dir=model_dir,
                          epochs=EPOCHS, learning_rate=LEARNING_RATE)
    with open(os.path.join(model_dir, "history.pkl"), 'wb') as f:
        pickle.dump(history.history, f)

    print(f"[평가 시작] {model_name}")
    best_model = load_model(os.path.join(model_dir, "best_model.keras"))
    loss, acc = best_model.evaluate(val_dataset, verbose=1)

    y_true, y_pred = [], []
    for batch_x, batch_y in val_dataset:
        preds = best_model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy())
        y_pred.extend(np.argmax(preds, 1))

    cm = confusion_matrix(y_true, y_pred)
    print(f"[DONE] {model_name}: acc={acc:.4f}, loss={loss:.4f}")

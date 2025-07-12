import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

class RadarDataset:
    def __init__(self, base_file_list, label_map, batch_size=32, shuffle=True, data_dir='dataset/preprocessed'):
        self.base_file_list = base_file_list
        self.label_map = label_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_dir = data_dir

        self.dataset = self._build_tf_dataset()

    def _load_paths_and_labels(self):
        rtm_paths, dtm_paths, labels = [], [], []
        for base in self.base_file_list:
            rtm_path = os.path.join(self.data_dir, f"{base}_rtm.npy")
            dtm_path = os.path.join(self.data_dir, f"{base}_dtm.npy")
            if os.path.exists(rtm_path) and os.path.exists(dtm_path):
                rtm_paths.append(rtm_path.encode())  # tf.numpy_function expects bytes
                dtm_paths.append(dtm_path.encode())
                labels.append(self.label_map[base])
        return np.array(rtm_paths), np.array(dtm_paths), np.array(labels)

    def _build_tf_dataset(self):
        rtm_paths, dtm_paths, labels = self._load_paths_and_labels()

        dataset = tf.data.Dataset.from_tensor_slices((rtm_paths, dtm_paths, labels))

        def _load_npy(rtm_path, dtm_path, label):
            rtm = tf.numpy_function(np.load, [rtm_path], tf.float32)
            rtm.set_shape([224,224,3])
            dtm = tf.numpy_function(np.load, [dtm_path], tf.float32)
            dtm.set_shape([224,224,3])
            return {"rtm_input": rtm, "dtm_input": dtm}, label

        dataset = dataset.map(_load_npy, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_dataset(self):
        return self.dataset

def get_class_weights(label_map, sample_list, num_classes):
    y = [label_map[sample] for sample in sample_list]
    weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y)
    return dict(enumerate(weights))

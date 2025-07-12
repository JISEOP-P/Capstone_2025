# train/train_model.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from utils.callback import CustomBestModelSaver, lr_schedule

def train_model(model, train_set, val_set, output_dir,
                class_weights=None, epochs=100, learning_rate=1e-3):
    
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.keras")

    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    callbacks = [
        CustomBestModelSaver(save_path=best_model_path),
        lr_scheduler
    ]

    history = model.fit(train_set,
                        validation_data=val_set,
                        epochs=epochs,
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=2
                        )

    return history  # 학습 곡선만 반환

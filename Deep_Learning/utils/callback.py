import numpy as np
from tensorflow.keras.callbacks import Callback

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
            print(f"[CUSTOM CHECKPOINT] Epoch {epoch+1}: 새로운 최고 정확도 {val_acc:.4f} - 모델 저장!")

        elif val_acc == self.best_acc and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save(self.save_path)
            print(f"[CUSTOM CHECKPOINT] Epoch {epoch+1}: 동률 정확도에서 낮은 손실 {val_loss:.4f} - 모델 저장!")

def lr_schedule(epoch, lr):
    if epoch == 25:
    # if epoch == 100:
        return lr * 0.1
    return lr
# export_to_onnx.py
import os
import tensorflow as tf
import tf2onnx
import onnx
from model.spectranet_v3 import build_spectranet_v3, DEFAULT_CFG
from tensorflow.keras import Model

NUM_CLASSES = 15
model = build_spectranet_v3(
    input_shapes={'rtm_input': (224,224,3), 'dtm_input': (224,224,3)},
    cfg=DEFAULT_CFG, num_classes=NUM_CLASSES
)

WEIGHTS = "experiments/Step_02/spectranet_v3/best_model.keras"
if os.path.exists(WEIGHTS):
    model.load_weights(WEIGHTS)
    print(f"[OK] loaded: {WEIGHTS}")

# ★ Softmax 제거
logits = model.get_layer("classifier").output
model_logits = Model(model.inputs, logits, name="SpectraNet_logits")

# ONNX 변환
spec = (tf.TensorSpec((None,224,224,3), tf.float32, name='rtm_input'),
        tf.TensorSpec((None,224,224,3), tf.float32, name='dtm_input'))

onnx_model, _ = tf2onnx.convert.from_keras(
    model_logits,
    input_signature=spec,
    opset=17,
    output_path="export/spectranet_logits.onnx"
)

onnx.checker.check_model("export/spectranet_logits.onnx")
print("[OK] export/spectranet_logits.onnx")

# model/spectranet.py
from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------------------------------
# Common
# -------------------------------------------------
DROPOUT_P = 0.1  # 모든 블록의 project-BN 뒤 고정 드롭아웃

def swish(x):
    return tf.nn.silu(x)

# EfficientNetV2 스타일 SE
# (입력 x는 이미 활성화가 적용된 텐서가 들어온다고 가정)
def SEBlock_EffV2(x, se_ratio=0.25, name=None):
    in_ch = x.shape[-1]
    se_ch = max(1, int(in_ch * se_ratio))
    # GAP → Reshape → Conv1x1(reduce) → Swish → Conv1x1(expand) → Sigmoid → Multiply
    s = layers.GlobalAveragePooling2D(name=None if not name else f"{name}_se_squeeze")(x)
    s = layers.Reshape((1, 1, in_ch), name=None if not name else f"{name}_se_reshape")(s)
    s = layers.Conv2D(se_ch, 1, padding='same', use_bias=True,
                      name=None if not name else f"{name}_se_reduce")(s)
    s = layers.Activation(swish, name=None if not name else f"{name}_se_swish")(s)
    s = layers.Conv2D(in_ch, 1, padding='same', use_bias=True,
                      name=None if not name else f"{name}_se_expand")(s)
    s = layers.Activation('sigmoid', name=None if not name else f"{name}_se_sigmoid")(s)
    return layers.Multiply(name=None if not name else f"{name}_se_excite")([x, s])

# -------------------------------------------------
# Blocks
# -------------------------------------------------
def MBConv(inputs, out_ch, stride=1, expand_ratio=4, use_se=True, name=None):
    """conv1x1(expand)-BN-Swish-Depthwise3x3-BN-Swish-SE?-conv1x1(project)-BN-Dropout-Add?"""
    in_ch = inputs.shape[-1]
    x = inputs
    mid_ch = int(in_ch * expand_ratio)

    # 1) expand 1x1
    x = layers.Conv2D(mid_ch, 1, padding='same', use_bias=False,
                      name=None if not name else f"{name}_expand")(x)
    x = layers.BatchNormalization(name=None if not name else f"{name}_expand_bn")(x)
    x = layers.Activation(swish, name=None if not name else f"{name}_expand_swish")(x)

    # 2) depthwise 3x3
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False,
                               name=None if not name else f"{name}_dw")(x)
    x = layers.BatchNormalization(name=None if not name else f"{name}_dw_bn")(x)
    x = layers.Activation(swish, name=None if not name else f"{name}_dw_swish")(x)

    # 3) SE (옵션)
    if use_se:
        x = SEBlock_EffV2(x, se_ratio=0.25, name=name)

    # 4) project 1x1
    x = layers.Conv2D(out_ch, 1, padding='same', use_bias=False,
                      name=None if not name else f"{name}_project")(x)
    x = layers.BatchNormalization(name=None if not name else f"{name}_project_bn")(x)

    # 5) dropout (고정 0.1)
    x = layers.Dropout(DROPOUT_P, name=None if not name else f"{name}_dropout")(x)

    # 6) skip (stride==1 & 채널 동일 시)
    if stride == 1 and in_ch == out_ch:
        x = layers.Add(name=None if not name else f"{name}_add")([x, inputs])
    return x


def FusedMBConv(inputs, out_ch, stride=1, expand_ratio=4, use_se=False, name=None):
    """conv2d(expand)-BN-Swish-SE?-conv1x1(project)-BN-Dropout-Add?"""
    in_ch = inputs.shape[-1]
    x = inputs
    mid_ch = int(in_ch * expand_ratio)  # t=1이면 mid=in_ch

    # 1) 3x3 conv (expand 포함, stride 반영)
    x = layers.Conv2D(mid_ch, 3, strides=stride, padding='same', use_bias=False,
                      name=None if not name else f"{name}_expand")(x)
    x = layers.BatchNormalization(name=None if not name else f"{name}_expand_bn")(x)
    x = layers.Activation(swish, name=None if not name else f"{name}_expand_swish")(x)

    # 2) SE (옵션)
    if use_se:
        x = SEBlock_EffV2(x, se_ratio=0.25, name=name)

    # 3) project 1x1
    x = layers.Conv2D(out_ch, 1, padding='same', use_bias=False,
                      name=None if not name else f"{name}_project")(x)
    x = layers.BatchNormalization(name=None if not name else f"{name}_project_bn")(x)

    # 4) dropout (고정 0.1)
    x = layers.Dropout(DROPOUT_P, name=None if not name else f"{name}_dropout")(x)

    # 5) skip (stride==1 & 채널 동일 시)
    if stride == 1 and in_ch == out_ch:
        x = layers.Add(name=None if not name else f"{name}_add")([x, inputs])
    return x

# -------------------------------------------------
# Stage/Stream/Model
# -------------------------------------------------
def _make_stage(x, cfg_stage: Dict[str, Any], prefix: str):
    block_type = cfg_stage['block_type']   # 'mbconv' or 'fused'
    out_ch     = cfg_stage['out_channels']
    stride     = cfg_stage['stride']
    expand     = cfg_stage['expand_ratio']
    use_se     = cfg_stage['use_se']
    repeats    = cfg_stage['repeats']

    for i in range(repeats):
        s = stride if i == 0 else 1
        name = f"{prefix}_s{out_ch}_b{i+1}"
        if block_type == 'mbconv':
            x = MBConv(x, out_ch, stride=s, expand_ratio=expand, use_se=use_se, name=name)
        else:
            x = FusedMBConv(x, out_ch, stride=s, expand_ratio=expand, use_se=use_se, name=name)
    return x

def build_single_stream(input_shape, stream_cfg: Dict[str, Any], stream_name: str):
    inp = layers.Input(shape=input_shape, name=f"{stream_name}_input")

    # Stem: Conv2D → BN → Swish (stride=2)
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False,
                      name=f"{stream_name}_stem_conv")(inp)
    x = layers.BatchNormalization(name=f"{stream_name}_stem_bn")(x)
    x = layers.Activation(swish, name=f"{stream_name}_stem_swish")(x)

    # Stage1~4
    for si in range(1, 5):
        x = _make_stage(x, stream_cfg[f"stage{si}"], prefix=f"{stream_name}_stage{si}")

    # Conv1x1 + GAP
    x = layers.Conv2D(stream_cfg.get("conv1x1_out", 256), 1, padding='same', use_bias=False,
                      name=f"{stream_name}_conv1x1")(x)
    x = layers.BatchNormalization(name=f"{stream_name}_conv1x1_bn")(x)
    x = layers.Activation(swish, name=f"{stream_name}_conv1x1_swish")(x)
    x = layers.GlobalAveragePooling2D(name=f"{stream_name}_gap")(x)
    return inp, x

def build_spectranet(input_shape=(224,224,3), cfg: Dict[str, Any]=None, num_classes=15):
    assert cfg is not None, "cfg를 제공하세요."
    rtm_in, rtm_feat = build_single_stream(input_shape, cfg['rtm'], 'rtm')
    dtm_in, dtm_feat = build_single_stream(input_shape, cfg['dtm'], 'dtm')

    merged = layers.Concatenate(name="merge_concat")([rtm_feat, dtm_feat])
    x = layers.Dropout(0.5)(merged)
    x = layers.Dense(64, name="head_fc")(x)
    x = layers.BatchNormalization(name="head_BN")(x)
    x = layers.Activation(swish, name="head_activation")(x)
    # 혼합정밀 안정성: 로짓/소프트맥스 float32 강제
    x = layers.Dense(num_classes, dtype='float32')(x)
    out = layers.Softmax(name='softmax', dtype='float32')(x)

    return Model(inputs=[rtm_in, dtm_in], outputs=out, name="SpectraNet")

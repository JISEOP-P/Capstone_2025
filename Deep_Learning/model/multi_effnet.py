from tensorflow.keras.layers import Input, Dense, Softmax, Dropout, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0

def build_backbone(input_shape, name='backbone'):
    input_tensor = Input(shape=input_shape)
    x = EfficientNetV2B0(
        include_top=False,
        weights=None,
        pooling='avg',
        include_preprocessing=False
    )(input_tensor)
    return Model(inputs=input_tensor, outputs=x, name=name)

def build_multi_effnet(input_shapes, num_classes=15):
    # rtm branch
    rtm_input = Input(shape=input_shapes['rtm_input'], name='rtm_input')
    rtm_backbone = build_backbone(input_shapes['rtm_input'], name='rtm_backbone')
    rtm_features = rtm_backbone(rtm_input)
    rtm_features = BatchNormalization()(rtm_features)

    # dtm branch
    dtm_input = Input(shape=input_shapes['dtm_input'], name='dtm_input')
    dtm_backbone = build_backbone(input_shapes['dtm_input'], name='dtm_backbone')
    dtm_features = dtm_backbone(dtm_input)
    dtm_features = BatchNormalization()(dtm_features)

    # Concatenate features
    merged = Concatenate()([rtm_features, dtm_features])

    # Classifier head
    x = Dropout(0.5)(merged)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Dense(num_classes)(x)
    output = Softmax()(x)

    model = Model(inputs=[rtm_input, dtm_input], outputs=output)
    return model

if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    # 입력 형태 정의
    input_shapes = {
        'rtm_input': (224, 224, 3),
        'dtm_input': (224, 224, 3),
    }

    # 모델 생성
    model = build_multi_effnet(input_shapes=input_shapes, num_classes=15)
    model.summary()

    # ----------------------------
    # 파라미터 수 계산
    # ----------------------------
    total_params = model.count_params()
    print("\n================ Parameter Info ================")
    print(f"Total params: {total_params:,}")

    # ----------------------------
    # FLOPs 계산 (TensorFlow Profiler)
    # ----------------------------
    # ConcreteFunction 생성
    concrete_func = tf.function(model).get_concrete_function(
        [tf.TensorSpec([1, *input_shapes['rtm_input']], tf.float32),
         tf.TensorSpec([1, *input_shapes['dtm_input']], tf.float32)]
    )

    # 그래프 객체를 명시적으로 전달해야 함 (TF2.13 이상)
    graph = concrete_func.graph

    # 프로파일 실행
    flops_info = profile(graph, options=ProfileOptionBuilder.float_operation())
    flops = flops_info.total_float_ops

    print("\n================ Computation Info ================")
    if flops >= 1e9:
        print(f"FLOPs : {flops / 1e9:.3f} GFLOPs (batch=1 기준)")
    elif flops >= 1e6:
        print(f"FLOPs : {flops / 1e6:.3f} MFLOPs (batch=1 기준)")
    else:
        print(f"FLOPs : {flops:.3f} FLOPs (batch=1 기준)")

    print("──────────────────────────────────────────────")
    print("SpectraNet successfully profiled ✅")
    print("──────────────────────────────────────────────")

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

def build_multi_effnet(input_shapes, num_classes=10):
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
    x = Dropout(0.8)(merged)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Dense(num_classes)(x)
    output = Softmax()(x)

    model = Model(inputs=[rtm_input, dtm_input], outputs=output)
    return model

# 테스트
if __name__ == '__main__':
    input_shapes = {
        'rtm_input': (224, 224, 3),
        'dtm_input': (224, 224, 3)
    }
    model = build_multi_effnet(input_shapes=input_shapes, num_classes=8)
    model.summary()

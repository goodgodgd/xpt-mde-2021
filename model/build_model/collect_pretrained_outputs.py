"""
Keras pretrained models are listed in
https://keras.io/applications/
"""
import tensorflow as tf
import tensorflow.keras.applications as tfapp
import json
import os.path as op
import settings

IMG_SHAPE = (256, 384, 3)
NASNET_SHAPE = (IMG_SHAPE[0] + 2, IMG_SHAPE[1] + 2, 3)
EXCEPTION_SHAPE = (IMG_SHAPE[0] + 6, IMG_SHAPE[1] + 6, 3)


def extract_scaled_layers():
    models = collect_models()
    layer_names = collect_layers(models)
    with open(op.dirname(op.abspath(__file__)) + '/scaled_layers.json', 'w') as fp:
        json.dump(layer_names, fp, separators=(',\n', ': '))


def collect_models():
    models = dict()
    models["MobileNetV2"] = tfapp.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["NASNetMobile"] = tfapp.NASNetMobile(input_shape=NASNET_SHAPE, include_top=False, weights='imagenet')
    models["DenseNet121"] = tfapp.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["ResNet50V2"] = tfapp.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["NASNetLarge"] = tfapp.NASNetLarge(input_shape=NASNET_SHAPE, include_top=False, weights='imagenet')
    # tobe tested
    models["VGG16"] = tfapp.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["Xception"] = tfapp.Xception(input_shape=EXCEPTION_SHAPE, include_top=False, weights='imagenet')
    # 2021.01 added
    models["EfficientNetB0"] = tfapp.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["EfficientNetB3"] = tfapp.EfficientNetB3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["EfficientNetB5"] = tfapp.EfficientNetB5(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    # omit non 2^n shape
    # models["InceptionV3"] = tfapp.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    # models["InceptionResNetV2"] = \
    #     tfapp.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    return models


def collect_layers(models, print_layer_shapes=True):
    print("\n\ncollect scaled layers")
    scaled_shapes = [(IMG_SHAPE[0]//sc, IMG_SHAPE[1]//sc) for sc in [2, 4, 8, 16, 32]]
    total_layers = dict()

    print("\nprint model summary in markdown table")
    print("model name | # paramters | # layers\n--- | --- | ---")
    for model_name, model in models.items():
        print(f"{model_name} | {model.count_params()/1000000:1.1f}M | {len(model.layers)}")

    print("\n\nprint layer information")
    for model_name, model in models.items():
        print("\n\n\n" + "%"*30)
        print(f"model name: {model_name}")
        layer_info = dict()

        print("="*30 + "\ncollect last layer names of selected scales")
        for layer_index, layer in enumerate(model.layers):
            if print_layer_shapes:
                print("layer info:", layer_index, layer.name, layer.output_shape, layer.input_shape)

            if ("input" in layer.name) or (len(layer.output_shape) < 4):
                continue

            out_height, out_width = layer.output_shape[1:3]
            for scid, (sc_height, sc_width) in enumerate(scaled_shapes):
                if (sc_height == out_height) and (sc_width == out_width) \
                    and ((("NASNet" in model_name) and ("activation" in layer.name))
                         or ("NASNet" not in model_name)):
                    layer_info[scid] = [layer_index, layer.name, sc_height, sc_width]

        print("="*30 + "\ncollected layer names")
        for scid, info in layer_info.items():
            print(f"scale id: {scid}, layer index: {info[0]}, name: {info[1]}, shape: {info[2:]}")

        layer_info = [info for info in layer_info.values()]
        total_layers[model_name] = layer_info

    return total_layers


if __name__ == "__main__":
    extract_scaled_layers()


# ==============================
# example of making a new model
# refer to: https://github.com/tensorflow/tensorflow/issues/33129
# models are defined here: ~/.pyenv/versions/vode_37/lib/python3.7/site-packages/keras_applications

def make_new_model():
    resnet50 = tfapp.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    midout = resnet50.get_layer('conv3_block4_out').output
    output = convolution(midout, 128, 3, 1, 'custom_output')
    model = tf.keras.Model(resnet50.input, outputs=output)
    model.summary()


def convolution(x, filters, kernel_size, strides, name):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                  padding="same", activation="relu",
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.025),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  name=name)(x)
    return conv


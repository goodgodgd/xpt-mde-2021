import tensorflow as tf
import tensorflow.keras.applications as tfapp
import json
import settings

IMG_SHAPE = (128, 384, 3)


def extract_scaled_layers():
    models = collect_models()
    layer_names = collect_layers(models)
    with open(settings.sub_package_path + '/scaled_layers.json', 'w') as fp:
        json.dump(layer_names, fp)


def collect_models():
    models = dict()
    models["MobileNetV2"] = tfapp.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["NASNetMobile"] = tfapp.NASNetMobile(input_shape=(130, 386, 3), include_top=False, weights='imagenet')
    models["DenseNet121"] = tfapp.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["VGG16"] = tfapp.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["Xception"] = tfapp.Xception(input_shape=(134, 390, 3), include_top=False, weights='imagenet')
    models["ResNet50V2"] = tfapp.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    models["NASNetLarge"] = tfapp.NASNetLarge(input_shape=(130, 386, 3), include_top=False, weights='imagenet')

    # omit non 2^n shape
    # models["InceptionV3"] = tfapp.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    # models["InceptionResNetV2"] = \
    #     tfapp.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    return models


def collect_layers(models, print_layer_shapes=True):
    print("\n\ncollect scaled layers")
    scaled_heights = (IMG_SHAPE[0]/2, IMG_SHAPE[0]/4, IMG_SHAPE[0]/8, IMG_SHAPE[0]/16, IMG_SHAPE[0]/32)
    scaled_heights = [int(sc) for sc in scaled_heights]
    total_layers = dict()

    print("\nprint model summary in markdown table")
    print("model name | # paramters | # layers\n--- | --- | ---")
    for model_name, model in models.items():
        print(f"{model_name} | {model.count_params()/1000000:1.1f}M | {len(model.layers)}")

    print("\n\nprint layer information")
    for model_name, model in models.items():
        print("\n\n\n" + "%"*30)
        print(f"model name: {model_name}")
        layer_names = ["anonymous"] * len(scaled_heights)

        print("="*30 + "\ncollect last layer names of selected scales")
        for layer in model.layers:
            if print_layer_shapes:
                print(layer.name, layer.output_shape, layer.input_shape)

            if "input" in layer.name:
                continue

            out_height = layer.output_shape[1]
            for scid, sc_height in enumerate(scaled_heights):
                if sc_height == out_height \
                    and ((("NASNet" in model_name) and ("activation" in layer.name))
                         or ("NASNet" not in model_name)):
                    layer_names[scid] = layer.name

        print("="*30 + "\ncollected layer names")
        for layer_name, height in zip(layer_names, scaled_heights):
            print(f"scaled height: {height}, layer name: {layer_name}")

        total_layers[model_name] = layer_names

    return total_layers


if __name__ == "__main__":
    extract_scaled_layers()


# ==============================
# example of making a new model

def convolution(x, filters, kernel_size, strides, name):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                  padding="same", activation="relu",
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.025),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  name=name)(x)
    return conv

# midout = resnet50.get_layer('conv3_block4_out').output
# output = convolution(midout, 128, 3, 1, 'custom_output')
# model = tf.keras.Model(resnet50.input, outputs=output)
# model.summary()

# refer to: https://github.com/tensorflow/tensorflow/issues/33129
# models are defined here: ~/.pyenv/versions/vode_37/lib/python3.7/site-packages/keras_applications


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow.keras.applications as tfapp
import json

import settings
from config import opts
import utils.util_funcs as uf
from utils.util_class import WrongInputException
import model.build_model.model_utils as mu
from model.build_model.model_base import NoResizingModel

PRETRAINED_MODELS = ["MobileNetV2", "NASNetMobile", "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge"]


class PretrainedModel(NoResizingModel):
    def __init__(self, batch_size, image_shape, snippet_len, model_name, pretrained_weight):
        super().__init__(batch_size, image_shape, snippet_len)
        weights = "imagenet" if pretrained_weight else None
        output_layers = self.read_output_layers(settings.sub_package_path + '/scaled_layers.json')
        self.base_model = self.build_base_model(model_name, weights, output_layers)
        self.output_layer_names = output_layers[model_name]

    def build_base_model(self, model_name, weights, output_layers):
        input_shape = (self.height, self.width, self.channel)
        input_img = layers.Input(shape=input_shape, batch_size=self.batch, name="image")
        if model_name == "MobileNetV2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            pproc_img = preprocess_input(input_img)
            model = tfapp.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
        elif model_name == "NASNetMobile":
            from tensorflow.keras.applications.nasnet import preprocess_input
            pproc_img = preprocess_input(input_img)
            assert self.height == 128
            pproc_img = tf.image.resize(pproc_img, size=(130, 386), method="bilinear")
            model = tfapp.NASNetMobile(input_shape=(134, 390, 3), include_top=False, weights=weights)
        elif model_name == "DenseNet121":
            from tensorflow.keras.applications.densenet import preprocess_input
            pproc_img = preprocess_input(input_img)
            model = tfapp.DenseNet121(input_shape=input_shape, include_top=False, weights=weights)
        elif model_name == "VGG16":
            from tensorflow.keras.applications.vgg16 import preprocess_input
            pproc_img = preprocess_input(input_img)
            model = tfapp.VGG16(input_shape=input_shape, include_top=False, weights=weights)
        elif model_name == "Xception":
            from tensorflow.keras.applications.xception import preprocess_input
            pproc_img = preprocess_input(input_img)
            assert self.height == 128
            pproc_img = tf.image.resize(pproc_img, size=(134, 390, 3), method="bilinear")
            model = tfapp.Xception(input_shape=(134, 390, 3), include_top=False, weights=weights)
        elif model_name == "Xception":
            from tensorflow.keras.applications.xception import preprocess_input
            pproc_img = preprocess_input(input_img)
            model = tfapp.Xception(input_shape=input_shape, include_top=False, weights=weights)
        elif model_name == "ResNet50V2":
            from tensorflow.keras.applications.resnet import preprocess_input
            pproc_img = preprocess_input(input_img)
            model = tfapp.ResNet50V2(input_shape=input_shape, include_top=False, weights=weights)
        elif model_name == "NASNetLarge":
            from tensorflow.keras.applications.nasnet import preprocess_input
            pproc_img = preprocess_input(input_img)
            assert self.height == 128
            pproc_img = tf.image.resize(pproc_img, size=(130, 386), method="bilinear")
            model = tfapp.NASNetLarge(input_shape=(130, 386, 3), include_top=False, weights=weights)
        else:
            raise WrongInputException("Wrong pretrained model name: " + model_name)

        # collect multi scale convolutional outputs
        out_layer_names = output_layers[model_name]
        outputs = []
        for layer_name in out_layer_names:
            layer = model.get_layer(layer_name)
            print(layer.name, layer.get_input_shape_at(0), layer.get_output_shape_at(0))
            outputs.append(layer.output)

        # create model with multi scale outputs
        multi_scale_model = tf.keras.Model(model.input, outputs, name=model_name + "_base")
        multi_scale_features = multi_scale_model(pproc_img)
        base_model = tf.keras.Model(input_img, multi_scale_features, name=model_name)
        return base_model

    def read_output_layers(self, filepath):
        with open(filepath, 'r') as fp:
            output_layers = json.load(fp)
        return output_layers

    def get_model(self):
        # TODO:
        return self.base_model


def test_build_model():
    model = PretrainedModel(opts.BATCH_SIZE, (opts.IM_HEIGHT, opts.IM_WIDTH, 3),
                            opts.SNIPPET_LEN, "NASNetMobile", opts.PRETRAINED_WEIGHT).get_model()
    model.summary()
    print("model output shapes:")
    for out in model.output:
        print(out.name, out.get_shape().as_list())


if __name__ == "__main__":
    test_build_model()


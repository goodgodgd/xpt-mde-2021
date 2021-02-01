import os.path as op
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.applications as tfapp
import json

from config import opts
from utils.util_class import WrongInputException


class PretrainedModel:
    def __init__(self, net_name, use_pt_weight):
        self.net_name = net_name
        self.pretrained_weight = use_pt_weight

    def encode(self, input_image):
        """
        :param input_image: (batch, height, width, channel)
        """
        input_shape = input_image.get_shape()[1:]
        height, width = input_shape[:2]
        net_name = self.net_name
        weights = "imagenet" if self.pretrained_weight else None

        jsonfile = op.join(opts.PROJECT_ROOT, "model", "build_model", "scaled_layers.json")
        output_layers = self.read_output_layers(jsonfile)
        out_layer_names = output_layers[net_name]
        nasnet_shape = (height + 2, width + 2, 3)
        xception_shape = (height + 6, width + 6, 3)

        if net_name == "MobileNetV2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_mobilenet")(input_image)
            ptmodel = tfapp.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "NASNetMobile":
            from tensorflow.keras.applications.nasnet import preprocess_input

            def preprocess_layer(x):
                x = preprocess_input(x)
                x = tf.image.resize(x, size=nasnet_shape[:2], method="bilinear")
                return x
            pproc_img = layers.Lambda(lambda x: preprocess_layer(x), name="preprocess_nasnet")(input_image)
            ptmodel = tfapp.NASNetMobile(input_shape=nasnet_shape, include_top=False, weights=weights)

        elif net_name == "DenseNet121":
            from tensorflow.keras.applications.densenet import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_densenet")(input_image)
            ptmodel = tfapp.DenseNet121(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "VGG16":
            from tensorflow.keras.applications.vgg16 import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_vgg16")(input_image)
            ptmodel = tfapp.VGG16(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "Xception":
            from tensorflow.keras.applications.xception import preprocess_input
            assert height == 128

            def preprocess_layer(x):
                x = preprocess_input(x)
                x = tf.image.resize(x, size=xception_shape[:2], method="bilinear")
                return x
            pproc_img = layers.Lambda(lambda x: preprocess_layer(x), name="preprocess_xception")(input_image)
            ptmodel = tfapp.Xception(input_shape=xception_shape, include_top=False, weights=weights)

        elif net_name == "ResNet50V2":
            from tensorflow.keras.applications.resnet import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_resnet")(input_image)
            ptmodel = tfapp.ResNet50V2(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "NASNetLarge":
            from tensorflow.keras.applications.nasnet import preprocess_input
            assert height == 128

            def preprocess_layer(x):
                x = preprocess_input(x)
                x = tf.image.resize(x, size=nasnet_shape[:2], method="bilinear")
                return x
            pproc_img = layers.Lambda(lambda x: preprocess_layer(x), name="preprocess_nasnet")(input_image)
            ptmodel = tfapp.NASNetLarge(input_shape=nasnet_shape, include_top=False, weights=weights)
        else:
            raise WrongInputException("Wrong pretrained model name: " + net_name)

        # collect multi scale convolutional features
        layer_outs = []
        for layer_name in out_layer_names:
            print(f"[PretrainedModel] {self.net_name}, output layer: {layer_name}")
            layer = ptmodel.get_layer(name=layer_name[1])
            # print("extract feature layers:", layer.name, layer.get_input_shape_at(0), layer.get_output_shape_at(0))
            layer_outs.append(layer.output)

        # create model with multi scale features
        multi_scale_model = tf.keras.Model(ptmodel.input, layer_outs, name=net_name + "_base")
        features_ms = multi_scale_model(pproc_img)
        return features_ms

    def read_output_layers(self, filepath):
        with open(filepath, 'r') as fp:
            output_layers = json.load(fp)
        return output_layers


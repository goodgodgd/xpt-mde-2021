import os.path as op
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.applications as tfapp
import json

import settings
from config import opts
from utils.util_class import WrongInputException
from model.build_model.model_base import DepthNetNoResize

NASNET_SHAPE = (130, 386, 3)
XCEPTION_SHAPE = (134, 390, 3)


class PretrainedModel:
    def __call__(self, input_tensor, total_shape, net_name, pretrained_weight):
        """
        :param input_tensor:
        :param total_shape:
        :param net_name:
        :param pretrained_weight:
        :return:
        """
        batch, snippet, height, width, channel = total_shape
        input_shape = (height, width, channel)
        weights = "imagenet" if pretrained_weight else None
        jsonfile = op.join(opts.PROJECT_ROOT, "model", "build_model", "scaled_layers.json")
        output_layers = self.read_output_layers(jsonfile)
        out_layer_names = output_layers[net_name]

        if net_name == "MobileNetV2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_mobilenet")(input_tensor)
            model = tfapp.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "NASNetMobile":
            from tensorflow.keras.applications.nasnet import preprocess_input
            assert height == 128

            def preprocess_layer(x):
                x = preprocess_input(x)
                x = tf.image.resize(x, size=NASNET_SHAPE[:2], method="bilinear")
                return x
            pproc_img = layers.Lambda(lambda x: preprocess_layer(x), name="preprocess_nasnet")(input_tensor)
            model = tfapp.NASNetMobile(input_shape=NASNET_SHAPE, include_top=False, weights=weights)

        elif net_name == "DenseNet121":
            from tensorflow.keras.applications.densenet import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_densenet")(input_tensor)
            model = tfapp.DenseNet121(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "VGG16":
            from tensorflow.keras.applications.vgg16 import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_vgg16")(input_tensor)
            model = tfapp.VGG16(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "Xception":
            from tensorflow.keras.applications.xception import preprocess_input
            assert height == 128

            def preprocess_layer(x):
                x = preprocess_input(x)
                x = tf.image.resize(x, size=XCEPTION_SHAPE[:2], method="bilinear")
                return x
            pproc_img = layers.Lambda(lambda x: preprocess_layer(x), name="preprocess_xception")(input_tensor)
            model = tfapp.Xception(input_shape=XCEPTION_SHAPE, include_top=False, weights=weights)

        elif net_name == "ResNet50V2":
            from tensorflow.keras.applications.resnet import preprocess_input
            pproc_img = layers.Lambda(lambda x: preprocess_input(x), name="preprocess_resnet")(input_tensor)
            model = tfapp.ResNet50V2(input_shape=input_shape, include_top=False, weights=weights)

        elif net_name == "NASNetLarge":
            from tensorflow.keras.applications.nasnet import preprocess_input
            assert height == 128

            def preprocess_layer(x):
                x = preprocess_input(x)
                x = tf.image.resize(x, size=NASNET_SHAPE[:2], method="bilinear")
                return x
            pproc_img = layers.Lambda(lambda x: preprocess_layer(x), name="preprocess_nasnet")(input_tensor)
            model = tfapp.NASNetLarge(input_shape=NASNET_SHAPE, include_top=False, weights=weights)
        else:
            raise WrongInputException("Wrong pretrained model name: " + net_name)

        # collect multi scale convolutional outputs
        outputs = []
        for layer_name in out_layer_names:
            layer = model.get_layer(layer_name)
            print(layer.name, layer.get_input_shape_at(0), layer.get_output_shape_at(0))
            outputs.append(layer.output)

        # create model with multi scale outputs
        multi_scale_model = tf.keras.Model(model.input, outputs, name=net_name + "_base")
        multi_scale_features = multi_scale_model(pproc_img)
        return multi_scale_features

    def read_output_layers(self, filepath):
        with open(filepath, 'r') as fp:
            output_layers = json.load(fp)
        return output_layers


class DecoderForPretrained(DepthNetNoResize):
    def __call__(self, features_ms, input_shape):
        """
        :param features_ms: [conv_s1, conv_s2, conv_s3, conv_s4]
                conv'n' denotes convolutional feature map spatially scaled by 1/2^n
                if input height is 128, heights of features are (64, 32, 16, 8, 4) repectively
        :param input_shape: input tensor size (batch, snippet, height, width, channel)
        :return:
        """
        conv1, conv2, conv3, conv4, conv5 = features_ms
        batch, snippet, height, width, channel = input_shape

        # decoder by upsampling
        upconv4 = self.upconv_with_skip_connection(conv5, conv4, 256, "dp_up4")             # 1/16
        upconv3 = self.upconv_with_skip_connection(upconv4, conv3, 128, "dp_up3")           # 1/8
        disp3, disp2_up = self.get_disp_vgg(upconv3, height // 4, width // 4, "dp_disp3")   # 1/8
        upconv2 = self.upconv_with_skip_connection(upconv3, conv2, 64, "dp_up2", disp2_up)  # 1/4
        disp2, disp1_up = self.get_disp_vgg(upconv2, height // 2, width // 2, "dp_disp2")   # 1/4
        upconv1 = self.upconv_with_skip_connection(upconv2, conv1, 32, "dp_up1", disp1_up)  # 1/2
        disp1, disp0_up = self.get_disp_vgg(upconv1, height, width, "dp_disp1")             # 1/2
        upconv0 = self.upconv_with_skip_connection(upconv1, disp0_up, 16, "dp_up0")         # 1
        disp0, disp_n1_up = self.get_disp_vgg(upconv0, height, width, "dp_disp0")           # 1

        return [disp0, disp1, disp2, disp3]


# ==================================================
import utils.util_funcs as uf


def test_build_model():
    total_shape = (opts.BATCH_SIZE, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
    raw_image_shape = (opts.IM_HEIGHT * opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    snippet_image = layers.Input(shape=raw_image_shape, batch_size=opts.BATCH_SIZE, name="input_image")
    source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                               name="split_stacked_image")(snippet_image)

    features = PretrainedModel()(target_image, total_shape, "NASNetMobile", True)
    model = tf.keras.Model(snippet_image, features, name="feature_model")
    model.summary()
    print("model output shapes:")
    for out in model.output:
        print(out.name, out.get_shape().as_list())


if __name__ == "__main__":
    test_build_model()


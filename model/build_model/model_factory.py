import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as uf
from model.build_model.model_base import DepthNetBasic, DepthNetNoResize, PoseNet
from utils.util_class import WrongInputException
from model.build_model.pretrained_models import PretrainedModel, DecoderForPretrained

PRETRAINED_MODELS = ["MobileNetV2", "NASNetMobile", "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge"]


class ModelFactory:
    def __init__(self, input_shape=None, net_names=None, pretrained_weight=None):
        # set defualt values from config
        default_shape = (opts.BATCH_SIZE, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
        self.input_shape = default_shape if input_shape is None else input_shape
        self.net_names = opts.NET_NAMES if net_names is None else net_names
        self.pretrained_weight = opts.PRETRAINED_WEIGHT if pretrained_weight is None else pretrained_weight

    def get_model(self):
        # prepare input tensor
        batch, snippet, height, width, channel = self.input_shape
        raw_image_shape = (height*snippet, width, channel)
        snippet_image = layers.Input(shape=raw_image_shape, batch_size=batch, name="input_image")
        source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                                   name="split_stacked_image")(snippet_image)

        # build prediction models
        predictions = dict()
        if "depth" in self.net_names:
            depth_ms = self.depth_net_factory(target_image, self.net_names["depth"])
            predictions.update(depth_ms)
        if "camera" in self.net_names:
            # TODO: add intrinsic output
            camera = self.camera_net_factory(self.net_names["camera"], snippet_image)
            predictions.update(camera)
        # TODO: add optical flow factory

        # create model
        model = tf.keras.Model(snippet_image, predictions)
        return model

    def depth_net_factory(self, target_image, net_name):
        if net_name == "DepthNetBasic":
            disp_ms = DepthNetBasic()(target_image, self.input_shape)
        elif net_name == "DepthNetNoResize":
            disp_ms = DepthNetNoResize()(target_image, self.input_shape)
        elif net_name in PRETRAINED_MODELS:
            features_ms = PretrainedModel()(target_image, self.input_shape, net_name, self.pretrained_weight)
            disp_ms = DecoderForPretrained()(features_ms, self.input_shape)
        else:
            raise WrongInputException("[depth_net_factory] wrong depth net name: " + net_name)
        return {"disp_ms": disp_ms}

    def camera_net_factory(self, net_name, snippet_image):
        if net_name == "PoseNet":
            pose = PoseNet()(snippet_image, self.input_shape)
        else:
            raise WrongInputException("[camera_net_factory] wrong pose net name: " + net_name)

        return {"pose": pose}


# ==================================================
import os.path as op
from tensorflow.keras.utils import plot_model


def test_build_model():
    model = ModelFactory().get_model()
    model.summary()
    print("model output shapes:")
    for name, output in model.output.items():
        if isinstance(output, list):
            for out in output:
                print(name, out.name, out.get_shape().as_list())
        else:
            print(name, output.name, output.get_shape().as_list())

    # record model architecture into files
    plot_model(model, to_file=op.join(opts.PROJECT_ROOT, "../model.png"), show_shapes=True)
    summary_file = op.join(opts.PROJECT_ROOT, "../summary.txt")
    with open(summary_file, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


if __name__ == "__main__":
    test_build_model()

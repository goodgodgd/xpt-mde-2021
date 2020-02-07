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
    def get_model(self, input_shape=None, net_names=None, pretrained_weight=None):
        # set defualt values from config
        default_shape = (opts.BATCH_SIZE, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
        input_shape = default_shape if input_shape is None else input_shape
        net_names = opts.NET_NAMES if net_names is None else net_names
        pretrained_weight = opts.PRETRAINED_WEIGHT if pretrained_weight is None else pretrained_weight

        # prepare input tensor
        batch, snippet, height, width, channel = input_shape
        raw_image_shape = (height*snippet, width, channel)
        snippet_image = layers.Input(shape=raw_image_shape, batch_size=batch, name="input_image")
        source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                                   name="split_stacked_image")(snippet_image)

        # build prediction models
        predictions = dict()
        if "depth" in net_names:
            depth_ms = self.depth_net_factory(target_image, input_shape, net_names["depth"], pretrained_weight)
            predictions.update(depth_ms)
        if "camera" in net_names:
            # TODO: add intrinsic output
            camera = self.camera_net_factory(net_names["camera"], snippet_image, input_shape)
            predictions.update(camera)
        # TODO: add optical flow factory

        # create model
        model = tf.keras.Model(snippet_image, predictions)
        return model

    def depth_net_factory(self, target_image, input_shape, net_name, pretrained_weight=True):
        if net_name == "DepthNetBasic":
            disp_ms = DepthNetBasic()(target_image, input_shape)
        elif net_name == "DepthNetNoResize":
            disp_ms = DepthNetNoResize()(target_image, input_shape)
        elif net_name in PRETRAINED_MODELS:
            features_ms = PretrainedModel()(target_image, input_shape, net_name, pretrained_weight)
            disp_ms = DecoderForPretrained()(features_ms, input_shape)
        return {"disp_ms": disp_ms}

    def camera_net_factory(self, net_name, snippet_image, input_shape):
        if net_name == "pose_only":
            pose = PoseNet()(snippet_image, input_shape)
        else:
            raise WrongInputException("[camera_net_factory] wrong net_name: " + net_name)

        return {"pose": pose}


# ==================================================

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


if __name__ == "__main__":
    test_build_model()

import numpy as np
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
    def __init__(self, input_shape=None, net_names=None, pretrained_weight=None, stereo=None):
        # set defualt values from config
        default_shape = (opts.BATCH_SIZE, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
        self.input_shape = default_shape if input_shape is None else input_shape
        self.net_names = opts.NET_NAMES if net_names is None else net_names
        self.pretrained_weight = opts.PRETRAINED_WEIGHT if pretrained_weight is None else pretrained_weight
        self.stereo = opts.STEREO if stereo is None else stereo

    def get_model(self):
        # prepare input tensor
        batch, snippet, height, width, channel = self.input_shape
        raw_image_shape = (height*snippet, width, channel)
        stacked_image = layers.Input(shape=raw_image_shape, batch_size=batch, name="input_image")
        source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                                   name="split_stacked_image")(stacked_image)
        # build prediction models
        outputs = dict()
        if "depth" in self.net_names:
            depth_ms = self.depth_net_factory(target_image, self.net_names["depth"])
            outputs.update(depth_ms)
        if "camera" in self.net_names:
            # TODO: add intrinsic output
            camera = self.camera_net_factory(self.net_names["camera"], stacked_image)
            outputs.update(camera)
        # TODO: add optical flow factory

        # create model
        model = tf.keras.Model(inputs=stacked_image, outputs=outputs)
        if self.stereo:
            model_wrapper = StereoModelWrapper(model)
        else:
            model_wrapper = ModelWrapper(model)
        return model_wrapper

    def depth_net_factory(self, target_image, net_name):
        if net_name == "DepthNetBasic":
            disp_ms, conv_ms = DepthNetBasic()(target_image, self.input_shape)
        elif net_name == "DepthNetNoResize":
            disp_ms, conv_ms = DepthNetNoResize()(target_image, self.input_shape)
        elif net_name in PRETRAINED_MODELS:
            features_ms = PretrainedModel()(target_image, self.input_shape, net_name, self.pretrained_weight)
            disp_ms, conv_ms = DecoderForPretrained()(features_ms, self.input_shape)
        else:
            raise WrongInputException("[depth_net_factory] wrong depth net name: " + net_name)
        return {"disp_ms": disp_ms, "dp_internal": conv_ms}

    def camera_net_factory(self, net_name, snippet_image):
        if net_name == "PoseNet":
            pose = PoseNet()(snippet_image, self.input_shape)
        else:
            raise WrongInputException("[camera_net_factory] wrong pose net name: " + net_name)
        return {"pose": pose}


class ModelWrapper:
    """
    tf.keras.Model output formats according to prediction methods
    1) preds = model(image_tensor) -> dict('disp_ms': disp_ms, 'pose': pose)
        disp_ms: list of [batch, height/scale, width/scale, 1]
        pose: [batch, num_src, 6]
    2) preds = model.predict(image_tensor) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    3) preds = model.predict({'image':, ...}) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, features):
        preds = self.model(features["image"])
        return preds

    def predict(self, dataset, total_steps):
        return self.predict_oneside(dataset, "image", total_steps)

    def predict_oneside(self, dataset, image_key, total_steps):
        print(f"===== start prediction from [{image_key}] key")
        predictions = {"disp": [], "pose": []}
        for step, features in enumerate(dataset):
            pred = self.model.predict(features[image_key])
            predictions["disp"].append(pred[0])
            predictions["pose"].append(pred[4])
            uf.print_progress_status(f"Progress: {step} / {total_steps}")
        print("")

        predictions["disp"] = np.concatenate(predictions["disp"], axis=0)
        disp = predictions["disp"]
        mask = (disp > 0)
        depth = np.zeros(disp.shape, dtype=np.float)
        depth[mask] = 1. / disp[mask]
        predictions["depth"] = depth
        predictions["pose"] = np.concatenate(predictions["pose"], axis=0)
        return predictions

    def compile(self, optimizer="sgd", loss="mean_absolute_error"):
        self.model.compile(optimizer=optimizer, loss=loss)

    def trainable_weights(self):
        return self.model.trainable_weights

    def layers(self):
        return self.model.layers

    def save_weights(self, ckpt_path):
        self.model.save_weights(ckpt_path)

    def load_weights(self, ckpt_path):
        self.model.load_weights(ckpt_path)


class StereoModelWrapper(ModelWrapper):
    def __init__(self, model):
        super(StereoModelWrapper, self).__init__(model)

    def __call__(self, features):
        preds = self.model(features["image"])
        preds_rig = self.model(features["image_R"])
        preds["disp_ms_R"] = preds_rig["disp_ms"]
        preds["pose_R"] = preds_rig["pose"]
        return preds

    def predict(self, dataset, total_steps):
        preds = self.predict_oneside(dataset, "image", total_steps)
        preds_rig = self.predict_oneside(dataset, "image_R", total_steps)
        preds["disp_R"] = preds_rig["disp"]
        preds["depth_R"] = preds_rig["depth"]
        preds["pose_R"] = preds_rig["pose"]
        return preds


# ==================================================
import os.path as op
from tensorflow.keras.utils import plot_model
import model.build_model.model_utils as mu


def test_build_model():
    model = ModelFactory(stereo=True).get_model()
    model.summary()
    print("model output shapes:")
    for name, output in model.output.items():
        if isinstance(output, list):
            for out in output:
                print(name, out.name, out.get_shape().as_list())
        else:
            print(name, output.name, output.get_shape().as_list())

    # record model architecture into text and image files
    plot_model(model, to_file=op.join(opts.PROJECT_ROOT, "../model.png"), show_shapes=True)
    summary_file = op.join(opts.PROJECT_ROOT, "../summary.txt")
    with open(summary_file, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def test_model_dict_inout():
    input1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input1")
    conv1 = mu.convolution(input1, 32, 5, strides=1, name="conv1a")
    conv1 = mu.convolution(conv1, 32, 5, strides=2, name="conv1b")
    conv1 = mu.convolution(conv1, 64, 5, strides=1, name="conv1c")

    input2 = layers.Input(shape=(50, 50, 3), batch_size=8, name="input2")
    conv2 = mu.convolution(input2, 32, 5, strides=1, name="conv2a")
    conv2 = mu.convolution(conv2, 32, 5, strides=2, name="conv2b")
    conv2 = mu.convolution(conv2, 64, 5, strides=1, name="conv2c")

    feature = layers.Input(shape=(10, 10), batch_size=8, name="input1")

    inputs = {"input1": input1, "input2": input2}
    outputs = {"output1": conv1, "output2": conv2}
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="double_model")
    model.compile()
    model.summary()

    tsinput1 = tf.random.uniform(shape=(8, 100, 100, 3), minval=0, maxval=1)
    tsinput2 = tf.random.uniform(shape=(8, 50, 50, 3), minval=0, maxval=1)
    tsfeature = tf.random.uniform(shape=(8, 10, 10), minval=0, maxval=1)

    print("===== dict input: NO problem -> dict output")
    tsinputs = {"input1": tsinput1, "input2": tsinput2}
    predictions = model(tsinputs)
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")

    print("===== list input: NO problem -> dict output")
    predictions = model([tsinput1, tsinput2])
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")

    print("===== dict input with REVERSE order: NO problem -> dict output")
    tsinputs = {"input2": tsinput2, "input1": tsinput1}
    predictions = model(tsinputs)
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")

    print("===== list input with REVERSE order: PROBLEM -> dict output")
    predictions = model([tsinput2, tsinput1])
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")
    print("first element is used as 'input1', second is used as 'input2'")


def test_name_scope():
    # with tf.name_scope("left") as scope:
    #     image1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="image")
    #
    # with tf.name_scope("right") as scope:
    #     image2 = layers.Input(shape=(100, 100, 3), batch_size=8, name="image")

    image1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="image")
    image2 = layers.Input(shape=(100, 100, 3), batch_size=8)
    print(image1.name, image2.name)
    output = tf.concat([image1, image2], axis=1)
    model = tf.keras.Model(inputs=[image1, image2], outputs=output)
    model.summary()


if __name__ == "__main__":
    # test_build_model()
    # test_model_dict_inout()
    test_name_scope()

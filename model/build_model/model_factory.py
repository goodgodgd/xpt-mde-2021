import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import glob

import settings
from config import opts
import utils.util_funcs as uf
from model.build_model.model_base import DepthNetBasic, DepthNetNoResize, PoseNet
from utils.util_class import WrongInputException
from model.build_model.pretrained_models import PretrainedModel

PRETRAINED_MODELS = ["MobileNetV2", "NASNetMobile", "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge"]


class ModelFactory:
    def __init__(self, input_shape=None, net_names=None, pretrained_weight=None, stereo=None, stereo_extrinsic=None):
        # set defualt values from config
        default_shape = (opts.BATCH_SIZE, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
        self.input_shape = default_shape if input_shape is None else input_shape
        self.net_names = opts.NET_NAMES if net_names is None else net_names
        self.pretrained_weight = opts.PRETRAINED_WEIGHT if pretrained_weight is None else pretrained_weight
        self.stereo = opts.STEREO if stereo is None else stereo
        self.stereo_extrinsic = opts.STEREO_EXTRINSIC if stereo_extrinsic is None else stereo_extrinsic

    def get_model(self):
        # prepare input tensor
        batch, snippet, height, width, channel = self.input_shape
        raw_image_shape = (height*snippet, width, channel)
        stacked_image = layers.Input(shape=raw_image_shape, batch_size=batch, name="input_image")
        source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                                   name="split_stacked_image")(stacked_image)
        # build prediction models
        outputs = dict()
        other_image = None

        models = dict()
        if "depth" in self.net_names:
            depthnet = self.depth_net_factory(self.net_names["depth"])
            models["depthnet"] = depthnet

        if "camera" in self.net_names:
            # TODO: add intrinsic output
            posenet = self.camera_net_factory(self.net_names["camera"])
            models["posenet"] = posenet
        # TODO: add optical flow factory

        if self.stereo_extrinsic:
            model_wrapper = StereoPoseModelWrapper(models)
        elif self.stereo:
            model_wrapper = StereoModelWrapper(models)
        else:
            model_wrapper = ModelWrapper(models)

        return model_wrapper

    def depth_net_factory(self, net_name):
        if net_name == "DepthNetBasic":
            depth_net = DepthNetBasic()(self.input_shape)
        elif net_name == "DepthNetNoResize":
            depth_net = DepthNetNoResize()(self.input_shape)
        elif net_name in PRETRAINED_MODELS:
            depth_net = PretrainedModel()(self.input_shape, net_name, self.pretrained_weight)
        else:
            raise WrongInputException("[depth_net_factory] wrong depth net name: " + net_name)
        return depth_net
        # return {"disp_ms": disp_ms, "debug_out": debug_out}

    def camera_net_factory(self, net_name):
        if net_name == "PoseNet":
            posenet = PoseNet()(self.input_shape)
        else:
            raise WrongInputException("[camera_net_factory] wrong pose net name: " + net_name)
        return posenet
        # return {"pose": pose}


class ModelWrapper:
    """
    tf.keras.Model output formats according to prediction methods
    1) preds = model(image_tensor) -> dict('disp_ms': disp_ms, 'pose': pose)
        disp_ms: list of [batch, height/scale, width/scale, 1]
        pose: [batch, num_src, 6]
    2) preds = model.predict(image_tensor) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    3) preds = model.predict({'image':, ...}) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    """
    def __init__(self, models):
        self.models = models

    def __call__(self, features):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image"])
            predictions.update(pred)
        return predictions

    def predict(self, dataset, total_steps):
        return self.predict_oneside(dataset, "image", total_steps)

    def predict_oneside(self, dataset, image_key, total_steps):
        print(f"===== start prediction from [{image_key}] key")
        predictions = {"disp": [], "pose": []}
        for step, features in enumerate(dataset):
            disp_ms = self.models["depthnet"](features[image_key])
            predictions["disp"].append(disp_ms[0])
            pose = self.models["posenet"](features[image_key])
            predictions["pose"].append(pose)
            uf.print_progress_status(f"Progress: {step} / {total_steps}")

        print("")
        disp = np.concatenate(predictions["disp"], axis=0)
        predictions["disp"] = disp
        mask = (disp > 0)
        depth = np.zeros(disp.shape, dtype=np.float)
        depth[mask] = 1. / disp[mask]
        predictions["depth"] = depth
        predictions["pose"] = np.concatenate(predictions["pose"], axis=0)
        return predictions

    def compile(self, optimizer="sgd", loss="mean_absolute_error"):
        for model in self.models.values():
            model.compile(optimizer=optimizer, loss=loss)

    def trainable_weights(self):
        train_weights = []
        for model in self.models.values():
            train_weights.extend(model.trainable_weights)
        return train_weights

    def save_weights(self, ckpt_dir_path, suffix):
        for netname, model in self.models.items():
            save_path = op.join(ckpt_dir_path, f"{netname}_{suffix}.h5")
            model.save_weights(save_path)

    def load_weights(self, ckpt_dir_path):
        pattern = op.join(ckpt_dir_path, "*.h5")
        files = glob.glob(pattern)
        for ckpt_file in files:
            filename = op.basename(ckpt_file)
            netname = filename.split("_")[0]
            self.models[netname].load_weights(ckpt_file)

    def summary(self, **kwargs):
        for model in self.models.values():
            model.summary(**kwargs)

    def inputs(self):
        return [model.input for model in self.models.values()]

    def outputs(self):
        output_dict = dict()
        for model in self.models.values():
            output_dict.update(model.output)
        return output_dict

    def plot_model(self, dir_path):
        for netname, model in self.models.items():
            plot_model(model, to_file=op.join(dir_path, netname + ".png"), show_shapes=True)


class StereoModelWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, features):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image"])
            predictions.update(pred)
            preds_right = model(features["image_R"])
            preds_right = {key + "_R": value for key, value in preds_right.items()}
            predictions.update(preds_right)
        return predictions

    def predict(self, dataset, total_steps):
        predictions = self.predict_oneside(dataset, "image", total_steps)
        preds_right = self.predict_oneside(dataset, "image_R", total_steps)
        preds_right = {key + "_R": value for key, value in preds_right.items()}
        predictions.update(preds_right)
        return predictions


class StereoPoseModelWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def __call__(self, features):
        predictions = dict()
        for netname, model in self.models.items():
            pred = model(features["image"])
            predictions.update(pred)
            preds_right = model(features["image_R"])
            preds_right = {key + "_R": value for key, value in preds_right.items()}
            predictions.update(preds_right)

        # predicts stereo extrinsic in both directions: left to right, right to left
        if "posenet" in self.models:
            posenet = self.models["posenet"]
            left_source, left_target = uf.split_into_source_and_target(features["image"])
            right_source, right_target = uf.split_into_source_and_target(features["image_R"])
            num_src = opts.SNIPPET_LEN - 1
            lr_input = layers.concatenate([right_target] * num_src + [left_target], axis=1)
            rl_input = layers.concatenate([left_target] * num_src + [right_target], axis=1)
            # pose that transforms points from right to left (T_LR)
            pose_lr = posenet(lr_input)
            # pose that transforms points from left to right (T_RL)
            pose_rl = posenet(rl_input)
            predictions["pose_LR"] = pose_lr["pose"]
            predictions["pose_RL"] = pose_rl["pose"]

        return predictions


# ==================================================
import os.path as op
from tensorflow.keras.utils import plot_model
import model.build_model.model_utils as mu


def test_build_model():
    vode_model = ModelFactory(stereo=True).get_model()
    vode_model.summary()
    print("model input shapes:")
    for i, input_tensor in enumerate(vode_model.inputs()):
        print("input", i, input_tensor.name, input_tensor.get_shape())

    print("model output shapes:")
    for name, output in vode_model.outputs().items():
        if isinstance(output, list):
            for out in output:
                print(name, out.name, out.get_shape())
        else:
            print(name, output.name, output.get_shape())

    # record model architecture into text and image files
    vode_model.plot_model(op.dirname(opts.PROJECT_ROOT))
    summary_file = op.join(opts.PROJECT_ROOT, "../summary.txt")
    with open(summary_file, 'w') as fh:
        vode_model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("trainable weights", type(vode_model.trainable_weights()), len(vode_model.trainable_weights()))


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


def test_hierarchy_model():
    input1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input1")
    conv1 = mu.convolution(input1, 32, 5, strides=1, name="conv1a")
    conv1 = mu.convolution(conv1, 32, 5, strides=2, name="conv1b")
    conv1 = mu.convolution(conv1, 64, 5, strides=1, name="conv1c")
    model1 = tf.keras.Model(inputs=input1, outputs=conv1, name="model1")

    input2 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input2")
    conv2 = mu.convolution(input2, 32, 5, strides=1, name="conv2a")
    conv2 = mu.convolution(conv2, 64, 5, strides=2, name="conv2b")
    conv2 = mu.convolution(conv2, 32, 5, strides=1, name="conv2c")
    model2 = tf.keras.Model(inputs=input2, outputs=conv2, name="model2")

    input3 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input3")
    output1 = model1(input3)
    output2 = model2(input3)
    model = tf.keras.Model(inputs=input3, outputs={"out1": output1, "out2": output2}, name="higher_model")
    model.summary()


if __name__ == "__main__":
    test_build_model()
    # test_model_dict_inout()
    # test_name_scope()
    # test_hierarchy_model()

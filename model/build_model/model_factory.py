import tensorflow as tf

import settings
from config import opts
from utils.util_class import WrongInputException
import utils.util_funcs as uf
from model.build_model.depth_net import DepthNetBasic, DepthNetNoResize, DepthNetFromPretrained
from model.build_model.pose_net import PoseNet
from model.build_model.flow_net import PWCNet
import model.build_model.model_wrappers as mw
import model.model_util.layer_ops as lo


PRETRAINED_MODELS = ["MobileNetV2", "NASNetMobile", "DenseNet121", "VGG16", "Xception", "ResNet50V2", "NASNetLarge"]


class ModelFactory:
    def __init__(self, dataset_cfg,
                 global_batch=opts.BATCH_SIZE,
                 net_names=opts.JOINT_NET,
                 depth_activation=opts.DEPTH_ACTIVATION,
                 pretrained_weight=opts.PRETRAINED_WEIGHT,
                 stereo=opts.STEREO):
        self.global_batch = global_batch
        self.dataset_cfg = dataset_cfg
        self.bshwc_shape = [global_batch] + dataset_cfg["imshape"]
        self.net_names = opts.JOINT_NET if net_names is None else net_names
        print("[ModelFactory] net names:", self.net_names)
        self.activation = depth_activation
        self.pretrained_weight = pretrained_weight
        self.stereo = stereo

    def get_model(self):
        models = dict()

        if "depth" in self.net_names:
            conv_depth = self.conv2d_factory(opts.DEPTH_CONV_ARGS)
            depth_activation = self.activation_factory(self.activation)
            depth_upsample_method = opts.DEPTH_UPSAMPLE_INTERP
            depthnet = self.depth_net_factory(self.net_names["depth"], conv_depth,
                                              depth_activation, depth_upsample_method)
            models["depthnet"] = depthnet

        if "camera" in self.net_names:
            conv_pose = self.conv2d_factory(opts.POSE_CONV_ARGS)
            posenet = self.pose_net_factory(self.net_names["camera"], conv_pose)
            models["posenet"] = posenet

        if "flow" in self.net_names:
            conv_flow = self.conv2d_factory(opts.FLOW_CONV_ARGS)
            flownet = self.flow_net_factory(self.net_names["flow"], conv_flow)
            models["flownet"] = flownet

        if ("stereo_T_LR" in self.dataset_cfg) and ("depth" in self.net_names):
            model_wrapper = mw.StereoPoseModelWrapper(models)
        elif ("image_R" in self.dataset_cfg) and self.stereo:
            model_wrapper = mw.StereoModelWrapper(models)
        else:
            model_wrapper = mw.ModelWrapper(models)

        return model_wrapper

    def activation_factory(self, activ_name):
        if activ_name == "InverseSigmoid":
            return InverseSigmoidActivation()
        elif activ_name == "Exponential":
            return ExponentialActivation()
        else:
            WrongInputException("[activation_factory] wrong activation name: " + activ_name)

    def conv2d_factory(self, src_args):
        # convert string arguments for tf.keras.layers.Conv2D to object arguments
        dst_args = {}
        key = "activation"
        if key in src_args:
            if src_args[key] == "leaky_relu":
                dst_args[key] = tf.keras.layers.LeakyReLU(src_args[key + "_param"])
            else:
                dst_args[key] = tf.keras.layers.ReLU()

        key = "kernel_initializer"
        if key in src_args:
            if src_args[key] == "truncated_normal":
                dst_args[key] = tf.keras.initializers.TruncatedNormal(stddev=src_args[key + "_param"])
            else:
                dst_args[key] = tf.keras.initializers.GlorotUniform()

        key = "kernel_regularizer"
        if key in src_args:
            if src_args[key] == "l2":
                dst_args[key] = tf.keras.regularizers.l2(src_args[key + "_param"])

        # change default arguments of Conv2D layer
        conv_layer = lo.CustomConv2D(**dst_args)
        return conv_layer

    def depth_net_factory(self, net_name, conv2d_d, pred_activ, upsample_interp):
        if net_name == "DepthNetBasic":
            depth_net = DepthNetBasic(self.bshwc_shape, self.global_batch, conv2d_d, pred_activ, upsample_interp)()
        elif net_name == "DepthNetNoResize":
            depth_net = DepthNetNoResize(self.bshwc_shape, self.global_batch, conv2d_d, pred_activ, upsample_interp)()
        elif net_name in PRETRAINED_MODELS:
            depth_net = DepthNetFromPretrained(self.bshwc_shape, self.global_batch, conv2d_d, pred_activ, upsample_interp,
                                               net_name, self.pretrained_weight)()
        else:
            raise WrongInputException("[depth_net_factory] wrong depth net name: " + net_name)
        return depth_net

    def pose_net_factory(self, net_name, conv2d_p):
        if net_name == "PoseNet":
            posenet = PoseNet(self.bshwc_shape, self.global_batch, conv2d_p)()
        else:
            raise WrongInputException("[pose_net_factory] wrong pose net name: " + net_name)
        return posenet

    def flow_net_factory(self, net_name, conv2d_f):
        if net_name == "PWCNet":
            flownet = PWCNet(self.bshwc_shape, self.global_batch, conv2d_f)()
        else:
            raise WrongInputException("[flow_net_factory] wrong flow net name: " + net_name)
        return flownet


class InverseSigmoidActivation:
    def __call__(self, x):
        y = tf.math.sigmoid(x) + 0.01
        y = uf.safe_reciprocal_number(y)
        return y


class ExponentialActivation:
    def __call__(self, x):
        y = tf.math.sigmoid(x + 1.)*10. - 5.
        y = tf.exp(y)
        return y


# ==================================================
import os.path as op
from config import opts
from tfrecords.tfrecord_reader import TfrecordReader


def test_build_model():
    print("\n===== start test_build_model")
    vode_model = ModelFactory(stereo=True).get_model()
    vode_model.summary()
    print("model input shapes:")
    for i, input_tensor in enumerate(vode_model.inputs()):
        print("input", i, input_tensor.name, input_tensor.get_shape())

    print_dict_tensor_shape(vode_model.outputs(), "model output")

    # record model architecture into text and image files
    vode_model.plot_model(op.dirname(opts.PROJECT_ROOT))
    summary_file = op.join(opts.PROJECT_ROOT, "../summary.txt")
    with open(summary_file, 'w') as fh:
        vode_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    print("!!! test_build_model passed")


def test_model_predictions():
    print("\n===== start test_model_predictions")
    tfrgen = TfrecordReader(op.join(opts.DATAPATH_TFR, "kitti_raw_test"), shuffle=False)
    dataset = tfrgen.get_dataset()
    total_steps = tfrgen.get_total_steps()
    vode_model = ModelFactory(stereo=True).get_model()

    print("----- model wrapper __call__() output")
    for bi, features in enumerate(dataset):
        print("\nbatch index:", bi)
        outputs = vode_model(features)
        print_dict_tensor_shape(outputs, "output")
        if bi > 0:
            break

    print("\n----- model wrapper predict() output")
    predictions = vode_model.predict(dataset, total_steps)
    print_dict_tensor_shape(predictions, "predict")

    print("!!! test_model_predictions passed")


def print_dict_tensor_shape(dictdata, title):
    for name, data in dictdata.items():
        if isinstance(data, list):
            for datum in data:
                print(f"{title}: key={name}, shape={datum.get_shape()}")
        else:
            print(f"{title}: key={name}, shape={data.get_shape()}")


if __name__ == "__main__":
    # test_build_model()
    test_model_predictions()


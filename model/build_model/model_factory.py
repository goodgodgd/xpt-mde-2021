import settings
from config import opts
from utils.util_class import WrongInputException
import model.build_model.nets as nets
from model.build_model.pretrained_nets import PretrainedModel
import model.build_model.model_wrappers as mw


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
            model_wrapper = mw.StereoPoseModelWrapper(models)
        elif self.stereo:
            model_wrapper = mw.StereoModelWrapper(models)
        else:
            model_wrapper = mw.ModelWrapper(models)

        return model_wrapper

    def depth_net_factory(self, net_name):
        if net_name == "DepthNetBasic":
            depth_net = nets.DepthNetBasic()(self.input_shape)
        elif net_name == "DepthNetNoResize":
            depth_net = nets.DepthNetNoResize()(self.input_shape)
        elif net_name in PRETRAINED_MODELS:
            depth_net = PretrainedModel()(self.input_shape, net_name, self.pretrained_weight)
        else:
            raise WrongInputException("[depth_net_factory] wrong depth net name: " + net_name)
        return depth_net

    def camera_net_factory(self, net_name):
        if net_name == "PoseNet":
            posenet = nets.PoseNet()(self.input_shape)
        else:
            raise WrongInputException("[camera_net_factory] wrong pose net name: " + net_name)
        return posenet


# ==================================================
import os.path as op


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


if __name__ == "__main__":
    test_build_model()

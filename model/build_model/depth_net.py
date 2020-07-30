import tensorflow as tf
from tensorflow.keras import layers

import settings
import utils.util_funcs as uf
import model.model_util.layer_ops as lo
from model.build_model.pretrained_nets import PretrainedModel


class DepthNetBasic:
    """
    Basic DepthNet model used in sfmlearner and geonet
    """
    def __init__(self, total_shape, conv2d, pred_depth, upsample_iterp):
        """
        :param total_shape: explicit shape (batch, snippet, height, width, channel)
        :param conv2d: 2d convolution operator
        :param pred_depth: final activation function for depth prediction
        :param upsample_iterp: upsampling method
        """
        self.total_shape = total_shape
        self.depth_activation = pred_depth
        self.conv2d_d = conv2d
        self.upsample_method = upsample_iterp
        print("[DepthNet] convolution default options:", vars(conv2d))

    def __call__(self):
        """
        In the code below, the 'n' in conv'n' or upconv'n' represents scale of the feature map
        conv'n' implies that it is scaled by 1/2^n
        """
        batch, snippet, height, width, channel = self.total_shape
        input_shape = (snippet, height, width, channel)
        input_tensor = layers.Input(shape=input_shape, batch_size=batch, name="depthnet_input")
        target_image = layers.Lambda(lambda image: image[:, -1], name="depthnet_target")(input_tensor)

        convs = self.encode(target_image)
        outputs = self.decode(convs)

        depthnet = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="depthnet")
        return depthnet

    def encode(self, image):
        conv0 = self.conv2d_d(image, 32, 7, strides=1, name="dp_conv0b")
        conv1 = self.conv2d_d(conv0, 32, 7, strides=2, name="dp_conv1a")
        conv1 = self.conv2d_d(conv1, 64, 5, strides=1, name="dp_conv1b")
        conv2 = self.conv2d_d(conv1, 64, 5, strides=2, name="dp_conv2a")
        conv2 = self.conv2d_d(conv2, 128, 3, strides=1, name="dp_conv2b")
        conv3 = self.conv2d_d(conv2, 128, 3, strides=2, name="dp_conv3a")
        conv3 = self.conv2d_d(conv3, 256, 3, strides=1, name="dp_conv3b")
        conv4 = self.conv2d_d(conv3, 256, 3, strides=2, name="dp_conv4a")
        conv4 = self.conv2d_d(conv4, 512, 3, strides=1, name="dp_conv4b")
        conv5 = self.conv2d_d(conv4, 512, 3, strides=2, name="dp_conv5a")
        conv5 = self.conv2d_d(conv5, 512, 3, strides=1, name="dp_conv5b")
        conv6 = self.conv2d_d(conv5, 512, 3, strides=2, name="dp_conv6a")
        conv6 = self.conv2d_d(conv6, 512, 3, strides=1, name="dp_conv6b")
        conv7 = self.conv2d_d(conv6, 512, 3, strides=2, name="dp_conv7a")
        return [conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7]

    def decode(self, convs):
        conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7 = convs
        upconv6 = self.upconv_with_skip_connection(conv7, [conv6], 512, "dp_up6")     # 1/64
        upconv5 = self.upconv_with_skip_connection(upconv6, [conv5], 512, "dp_up5")   # 1/32
        upconv4 = self.upconv_with_skip_connection(upconv5, [conv4], 256, "dp_up4")   # 1/16

        upconv3 = self.upconv_with_skip_connection(upconv4, [conv3], 128, "dp_up3")   # 1/8
        dpconv3, depth3 = self.predict_depth(upconv3, "dp_depth3")

        dpconv2_up = resize_like(dpconv3, conv2, "dp_resize2")
        upconv2 = self.upconv_with_skip_connection(upconv3, [conv2, dpconv2_up], 64, "dp_up2")  # 1/4
        dpconv2, depth2 = self.predict_depth(upconv2, "dp_depth2")

        dpconv1_up = resize_like(dpconv2, conv1, "dp_resize1")
        upconv1 = self.upconv_with_skip_connection(upconv2, [conv1, dpconv1_up], 32, "dp_up1")  # 1/2
        dpconv1, depth1 = self.predict_depth(upconv1, "dp_depth1")

        dpconv0_up = resize_like(dpconv1, conv0, "dp_resize0")
        upconv0 = self.upconv_with_skip_connection(upconv1, [dpconv0_up], 16, "dp_up0")         # 1
        dpconv0, depth0 = self.predict_depth(upconv0, "dp_depth0")

        outputs = {"depth_ms": [depth0, depth1, depth2, depth3],
                   "debug_out": [upconv0, upconv3]}
        return outputs

    def upconv_with_skip_connection(self, bef_layer, skip_layer: list, out_channels: int, name: str):
        return UpconvWithSkip(self.conv2d_d, out_channels, self.upsample_method, name)([bef_layer, skip_layer])

    def predict_depth(self, src, scope):
        print("predict depth", scope, src.get_shape(), src.dtype)
        conv = self.conv2d_d(src, 1, 3, activation="linear", name=scope + "_conv")
        depth = layers.Lambda(lambda x: self.depth_activation(x), name=scope + "_acti")(conv)
        return conv, depth


class DepthNetNoResize(DepthNetBasic):
    def __init__(self, total_shape, conv2d, pred_depth, upsample_iterp):
        super().__init__(total_shape, conv2d, pred_depth, upsample_iterp)
    """
    Modified BasicModel to remove resizing features in decoding layers
    Width and height of input image must be integer multiple of 128
    """
    def upconv_with_skip_connection(self, bef_layer, skip_layer: list, out_channels: int, name: str):
        return UpconvNoResize(self.conv2d_d, out_channels, self.upsample_method, name)([bef_layer, skip_layer])


class DepthNetFromPretrained(DepthNetNoResize):
    def __init__(self, total_shape, conv2d, pred_depth, upsample_iterp, net_name, use_pt_weight):
        """
        :param total_shape: explicit shape (batch, snippet, height, width, channel)
        :param conv2d: 2d convolution operator
        :param pred_depth: final activation function for depth prediction
        :param upsample_iterp: upsampling method
        :param net_name: pretrained model name
        :param use_pt_weight: whether use pretrained weights
        """
        super().__init__(total_shape, conv2d, pred_depth, upsample_iterp)
        self.net_name = net_name
        self.pretrained_weight = use_pt_weight

    def encode(self, image):
        convs = PretrainedModel(self.net_name, self.pretrained_weight).encode(image)
        convs = [image] + convs
        return convs

    def decode(self, convs):
        """
        :param convs: [conv_s1, conv_s2, conv_s3, conv_s4]
                conv'n' denotes convolutional feature map spatially scaled by 1/2^n
                if input height is 128, heights of features are (64, 32, 16, 8, 4) repectively
        """
        image, conv1, conv2, conv3, conv4, conv5 = convs

        upconv4 = self.upconv_with_skip_connection(conv5, [conv4], 256, "dp_up4")   # 1/16

        upconv3 = self.upconv_with_skip_connection(upconv4, [conv3], 128, "dp_up3")   # 1/8
        dpconv3, depth3 = self.predict_depth(upconv3, "dp_depth3")

        dpconv2_up = resize_like(dpconv3, conv2, "dp_resize2")
        upconv2 = self.upconv_with_skip_connection(upconv3, [conv2, dpconv2_up], 64, "dp_up2")  # 1/4
        dpconv2, depth2 = self.predict_depth(upconv2, "dp_depth2")

        dpconv1_up = resize_like(dpconv2, conv1, "dp_resize1")
        upconv1 = self.upconv_with_skip_connection(upconv2, [conv1, dpconv1_up], 32, "dp_up1")  # 1/2
        dpconv1, depth1 = self.predict_depth(upconv1, "dp_depth1")

        dpconv0_up = resize_like(dpconv1, image, "dp_resize0")
        upconv0 = self.upconv_with_skip_connection(upconv1, [dpconv0_up], 16, "dp_up0")         # 1
        dpconv0, depth0 = self.predict_depth(upconv0, "dp_depth0")

        outputs = {"depth_ms": [depth0, depth1, depth2, depth3],
                   "debug_out": [dpconv0, upconv0, dpconv3, upconv3]}
        return outputs


class UpconvWithSkip(layers.Layer):
    def __init__(self, conv2d, out_channels, upsample_method, name):
        super().__init__(name=name)
        self.conv2d = conv2d
        self.out_channels = out_channels
        self.upsample_2x = layers.UpSampling2D(size=(2, 2), interpolation=upsample_method)

    def call(self, inputs, **kwargs):
        bef_layer, skip_layer = inputs
        return self.upconv_with_skip_connection(bef_layer, skip_layer)

    def upconv_with_skip_connection(self, bef_layer, skip_layer: list):
        upconv = self.upsample_2x(bef_layer)
        upconv = self.conv2d(upconv, self.out_channels, 3)
        upconv = resize_like(upconv, skip_layer[0])
        upconv = layers.Concatenate(axis=3)([upconv] + skip_layer)
        upconv = self.conv2d(upconv, self.out_channels, 3)
        return upconv


class UpconvNoResize(UpconvWithSkip):
    def __init__(self, conv2d, out_channels, upsample_method, name):
        super().__init__(conv2d, out_channels, upsample_method, name)

    def upconv_with_skip_connection(self, bef_layer, skip_layer: list):
        upconv = self.upsample_2x(bef_layer)
        upconv = self.conv2d(upconv, self.out_channels, 3)
        upconv = layers.Concatenate(axis=3)([upconv] + skip_layer)
        upconv = self.conv2d(upconv, self.out_channels, 3)
        return upconv


def resize_like(src, ref, op_name=None):
    height, width = ref.get_shape()[1:3]
    dst = layers.Lambda(lambda x: tf.image.resize(x, size=[height, width], method="bilinear"),
                        name=op_name)(src)
    return dst


# ==================================================
from config import opts
from model.model_util.layer_ops import CustomConv2D


class TempActivation:
    def __call__(self, x):
        y = tf.math.sigmoid(x) + 0.01
        y = uf.safe_reciprocal_number(y)
        return y


def test_build_model():
    total_shape = (opts.BATCH_SIZE, opts.SNIPPET_LEN, opts.IM_HEIGHT, opts.IM_WIDTH, 3)
    # depthnet = DepthNetBasic(total_shape, CustomConv2D(), TempActivation(), "nearest")()
    depthnet = DepthNetFromPretrained(total_shape, CustomConv2D(), TempActivation(),
                                      "nearest", "NASNetMobile", True)()
    depthnet.summary()
    depth_ms = depthnet.output["depth_ms"]
    debug_out = depthnet.output["debug_out"]
    print("multi scale depth outputs")
    for depth in depth_ms:
        print("\tdepth out", depth.name, depth.get_shape())
    print("depthnet outputs for debugging")
    for debug in debug_out:
        print("\tdebug out", debug.name, debug.get_shape())


if __name__ == "__main__":
    test_build_model()


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
    def __init__(self, total_shape, global_batch, conv2d, pred_depth, upsample_iterp, high_res):
        """
        :param total_shape: explicit shape (batch, snippet, height, width, channel)
        :param conv2d: 2d convolution operator
        :param pred_depth: final activation function for depth prediction
        :param upsample_iterp: upsampling method
        """
        self.total_shape = total_shape
        self.global_batch = global_batch
        self.predict_depth = pred_depth
        self.conv2d_d = conv2d
        self.upsample_interp_d = upsample_iterp
        self.high_res = high_res
        print("[DepthNet] convolution default options:", vars(conv2d))

    def __call__(self):
        """
        In the code below, the 'n' in conv'n' or upconv'n' represents scale of the feature map
        conv'n' implies that it is scaled by 1/2^n
        """
        batch, snippet, height, width, channel = self.total_shape
        input_shape = (snippet, height, width, channel)
        input_tensor = layers.Input(shape=input_shape, batch_size=self.global_batch, name="depthnet_input")
        target_image = layers.Lambda(lambda image: image[:, -1], name="depthnet_target")(input_tensor)

        conv0 = self.conv2d_d(target_image, 32, 7, strides=1, name="dp_conv0b")
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

        upconv6 = self.upconv_with_skip_connection(conv7, conv6, 512, "dp_up6")     # 1/64
        upconv5 = self.upconv_with_skip_connection(upconv6, conv5, 512, "dp_up5")   # 1/32
        upconv4 = self.upconv_with_skip_connection(upconv5, conv4, 256, "dp_up4")   # 1/16
        upconv3 = self.upconv_with_skip_connection(upconv4, conv3, 128, "dp_up3")   # 1/8
        depth3, dpconv2_up, dpconv3 = self.get_scaled_depth(upconv3, height // 4, width // 4, "dp_depth3")
        upconv2 = self.upconv_with_skip_connection(upconv3, conv2, 64, "dp_up2", dpconv2_up)  # 1/4
        depth2, dpconv1_up, dpconv2 = self.get_scaled_depth(upconv2, height // 2, width // 2, "dp_depth2")
        upconv1 = self.upconv_with_skip_connection(upconv2, conv1, 32, "dp_up1", dpconv1_up)  # 1/2
        depth1, dpconv0_up, dpconv1 = self.get_scaled_depth(upconv1, height, width, "dp_depth1")
        upconv0 = self.upconv_with_skip_connection(upconv1, dpconv0_up, 16, "dp_up0")         # 1
        depth0, dpconvn1_up, dpconv0 = self.get_scaled_depth(upconv0, height, width, "dp_depth0")

        outputs = {"depth_ms": [depth0, depth1, depth2, depth3],
                   "debug_out": [upconv0, upconv3]}
        depthnet = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="depthnet")
        return depthnet

    def upsample_2x_d(self, x, scope):
        # TODO: bilinear interpolation도 테스트
        upconv = layers.UpSampling2D(size=(2, 2), interpolation=self.upsample_interp_d, name=scope + "_sample")(x)
        return upconv

    def upconv_with_skip_connection(self, bef_layer, skip_layer, out_channels, scope, bef_pred=None):
        upconv = self.upsample_2x_d(bef_layer, scope)
        upconv = self.conv2d_d(upconv, out_channels, 3, name=scope + "_conv1")
        upconv = lo.resize_like(upconv, skip_layer, scope)
        upconv = tf.cond(bef_pred is not None,
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer, bef_pred]),
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer])
                         )
        upconv = self.conv2d_d(upconv, out_channels, 3, name=scope + "_conv2")
        return upconv

    def get_scaled_depth(self, src, dst_height, dst_width, scope):
        conv = self.conv2d_d(src, 1, 3, activation="linear", name=scope + "_conv")
        depth = layers.Lambda(lambda x: self.predict_depth(x), name=scope + "_acti")(conv)
        conv_up = lo.resize_image(conv, dst_height, dst_width, scope)
        return depth, conv_up, conv


class DepthNetNoResize(DepthNetBasic):
    def __init__(self, total_shape, global_batch, conv2d, pred_depth, upsample_iterp, high_res):
        super().__init__(total_shape, global_batch, conv2d, pred_depth, upsample_iterp, high_res)
    """
    Modified BasicModel to remove resizing features in decoding layers
    Width and height of input image must be integer multiple of 128
    """
    def upconv_with_skip_connection(self, bef_layer, skip_layer, out_channels, scope, bef_pred=None):
        upconv = self.upsample_2x_d(bef_layer, scope)
        upconv = self.conv2d_d(upconv, out_channels, 3, name=scope + "_conv1")
        upconv = tf.cond(bef_pred is not None,
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer, bef_pred]),
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer])
                         )
        upconv = self.conv2d_d(upconv, out_channels, 3, name=scope + "_conv2")
        return upconv


class DepthNetPretrained(DepthNetNoResize):
    def __init__(self, total_shape, global_batch, conv2d, pred_depth, upsample_iterp, net_name, use_pt_weight, high_res):
        """
        :param total_shape: explicit shape (batch, snippet, height, width, channel)
        :param conv2d: 2d convolution operator
        :param pred_depth: final activation function for depth prediction
        :param upsample_iterp: upsampling method
        :param net_name: pretrained model name
        :param use_pt_weight: whether use pretrained weights
        """
        super().__init__(total_shape, global_batch, conv2d, pred_depth, upsample_iterp, high_res)
        self.net_name = net_name
        self.pretrained = use_pt_weight

    def __call__(self):
        batch, snippet, height, width, channel = self.total_shape
        input_shape = (snippet, height, width, channel)
        input_tensor = layers.Input(shape=input_shape, batch_size=self.global_batch, name="depthnet_input")
        target_image = layers.Lambda(lambda input_t: input_t[:, -1], name="depthnet_target")(input_tensor)

        features_ms = PretrainedModel(self.net_name, self.pretrained).encode(target_image)
        outputs = self.decode(features_ms)
        depthnet = tf.keras.Model(inputs=input_tensor, outputs=outputs, name=self.net_name + "_base")
        return depthnet

    def decode(self, features_ms):
        """
        :param features_ms: [conv_s1, conv_s2, conv_s3, conv_s4]
                conv'n' denotes convolutional feature map spatially scaled by 1/2^n
                if input height is 128, heights of features are (64, 32, 16, 8, 4) respectively
        """
        conv1, conv2, conv3, conv4, conv5 = features_ms
        batch, snippet, height, width, channel = self.total_shape

        if self.high_res:
            conv6 = self.conv2d_d(conv5, 512, 3, strides=2, name="dp_conv6a")
            conv6 = self.conv2d_d(conv6, 512, 3, strides=1, name="dp_conv6b")
            conv6 = self.conv2d_d(conv6, 512, 3, strides=1, name="dp_conv6b")
            conv7 = self.conv2d_d(conv6, 512, 3, strides=2, name="dp_conv7a")
            # decoder by upsampling
            upconv6 = self.upconv_with_skip_connection(conv7, conv6, 512, "dp_up6")     # 1/64
            upconv5 = self.upconv_with_skip_connection(upconv6, conv5, 512, "dp_up5")   # 1/32
            upconv4 = self.upconv_with_skip_connection(upconv5, conv4, 256, "dp_up4")   # 1/16
        else:
            # decoder by upsampling
            upconv4 = self.upconv_with_skip_connection(conv5, conv4, 256, "dp_up4")     # 1/16

        upconv3 = self.upconv_with_skip_connection(upconv4, conv3, 128, "dp_up3")       # 1/8
        depth3, dpconv2_up, dpconv3 = self.get_scaled_depth(upconv3, height // 4, width // 4, "dp_depth3")   # 1/8
        upconv2 = self.upconv_with_skip_connection(upconv3, conv2, 64, "dp_up2", dpconv2_up)  # 1/4
        depth2, dpconv1_up, dpconv2 = self.get_scaled_depth(upconv2, height // 2, width // 2, "dp_depth2")   # 1/4
        upconv1 = self.upconv_with_skip_connection(upconv2, conv1, 32, "dp_up1", dpconv1_up)  # 1/2
        depth1, dpconv0_up, dpconv1 = self.get_scaled_depth(upconv1, height, width, "dp_depth1")    # 1/2
        upconv0 = self.upconv_with_skip_connection(upconv1, dpconv0_up, 16, "dp_up0")   # 1
        depth0, dpconvn1_up, dpconv0 = self.get_scaled_depth(upconv0, height, width, "dp_depth0")  # 1

        outputs = {"depth_ms": [depth0, depth1, depth2, depth3],
                   "debug_out": [dpconv0, upconv0, dpconv3, upconv3]}
        return outputs


# ==================================================
from config import opts


class TempActivation:
    def __call__(self, x):
        y = tf.math.sigmoid(x) + 0.01
        y = uf.safe_reciprocal_number(y)
        return y


def test_build_model():
    total_shape = opts.get_img_shape("BSHWC")
    depthnet = DepthNetPretrained(total_shape, TempActivation(), "NASNetMobile", True)()
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


import tensorflow as tf
from tensorflow.keras import layers
import utils.util_funcs as uf

import settings
import model.build_model.model_utils as mu

DISP_SCALING_VGG = 10


class DepthNetBasic:
    """
    Basic DepthNet model used in sfmlearner and geonet
    """
    def __init__(self, total_shape, activation):
        """
        :param total_shape: explicit shape (batch, snippet, height, width, channel)
        :param activation: depth activation function or functor
        """
        self.total_shape = total_shape
        self.activate_depth = activation

    def __call__(self):
        """
        In the code below, the 'n' in conv'n' or upconv'n' represents scale of the feature map
        conv'n' implies that it is scaled by 1/2^n
        """
        batch, snippet, height, width, channel = self.total_shape
        input_shape = (height*snippet, width, channel)
        input_tensor = layers.Input(shape=input_shape, batch_size=batch, name="depthnet_input")
        source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                                   name="depthnet_split_image")(input_tensor)

        conv0 = mu.convolution(target_image, 32, 7, strides=1, name="dp_conv0b")
        conv1 = mu.convolution(conv0, 32, 7, strides=2, name="dp_conv1a")
        conv1 = mu.convolution(conv1, 64, 5, strides=1, name="dp_conv1b")
        conv2 = mu.convolution(conv1, 64, 5, strides=2, name="dp_conv2a")
        conv2 = mu.convolution(conv2, 128, 3, strides=1, name="dp_conv2b")
        conv3 = mu.convolution(conv2, 128, 3, strides=2, name="dp_conv3a")
        conv3 = mu.convolution(conv3, 256, 3, strides=1, name="dp_conv3b")
        conv4 = mu.convolution(conv3, 256, 3, strides=2, name="dp_conv4a")
        conv4 = mu.convolution(conv4, 512, 3, strides=1, name="dp_conv4b")
        conv5 = mu.convolution(conv4, 512, 3, strides=2, name="dp_conv5a")
        conv5 = mu.convolution(conv5, 512, 3, strides=1, name="dp_conv5b")
        conv6 = mu.convolution(conv5, 512, 3, strides=2, name="dp_conv6a")
        conv6 = mu.convolution(conv6, 512, 3, strides=1, name="dp_conv6b")
        conv7 = mu.convolution(conv6, 512, 3, strides=2, name="dp_conv7a")

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
                   "debug_out": [dpconv0, upconv0, dpconv3, upconv3]}
        depthnet = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="depthnet")
        return depthnet

    def upconv_with_skip_connection(self, bef_layer, skip_layer, out_channels, scope, bef_pred=None):
        upconv = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=scope + "_sample")(bef_layer)
        upconv = mu.convolution(upconv, out_channels, 3, strides=1, name=scope + "_conv1")
        upconv = mu.resize_like(upconv, skip_layer, scope)
        upconv = tf.cond(bef_pred is not None,
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer, bef_pred]),
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer])
                         )
        upconv = mu.convolution(upconv, out_channels, 3, strides=1, name=scope + "_conv2")
        return upconv

    def get_scaled_depth(self, src, dst_height, dst_width, scope):
        conv = layers.Conv2D(1, 3, strides=1, padding="same", activation="linear", name=scope + "_conv")(src)
        depth = layers.Lambda(lambda x: self.activate_depth(x), name=scope + "_acti")(conv)
        # disp = layers.Lambda(lambda x: tf.math.sigmoid(x) + 0.01, name=scope + "_acti")(conv)
        # depth = uf.safe_reciprocal_number(disp)
        conv_up = mu.resize_image(conv, dst_height, dst_width, scope)
        return depth, conv_up, conv


class DepthNetNoResize(DepthNetBasic):
    def __init__(self, total_shape, activation):
        super().__init__(total_shape, activation)
    """
    Modified BasicModel to remove resizing features in decoding layers
    Width and height of input image must be integer multiple of 128
    """
    def upconv_with_skip_connection(self, bef_layer, skip_layer, out_channels, scope, bef_pred=None):
        upconv = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=scope + "_sample")(bef_layer)
        upconv = mu.convolution(upconv, out_channels, 3, strides=1, name=scope + "_conv1")
        upconv = tf.cond(bef_pred is not None,
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer, bef_pred]),
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer])
                         )
        upconv = mu.convolution(upconv, out_channels, 3, strides=1, name=scope + "_conv2")
        return upconv


class PoseNet:
    def __call__(self, input_shape):
        batch, snippet, height, width, channel = input_shape
        stacked_image_shape = (snippet*height, width, channel)
        snippet_image = layers.Input(shape=stacked_image_shape, batch_size=batch, name="posenet_input")
        channel_stack_image = layers.Lambda(lambda image: mu.restack_on_channels(image, snippet),
                                            name="channel_stack")(snippet_image)
        print("[PoseNet] channel stacked image shape=", channel_stack_image.get_shape())
        num_sources = snippet - 1

        conv1 = mu.convolution(channel_stack_image, 16, 7, 2, "vo_conv1")
        conv2 = mu.convolution(conv1, 32, 5, 2, "vo_conv2")
        conv3 = mu.convolution(conv2, 64, 3, 2, "vo_conv3")
        conv4 = mu.convolution(conv3, 128, 3, 2, "vo_conv4")
        conv5 = mu.convolution(conv4, 256, 3, 2, "vo_conv5")
        conv6 = mu.convolution(conv5, 256, 3, 2, "vo_conv6")
        conv7 = mu.convolution(conv6, 256, 3, 2, "vo_conv7")

        poses = tf.keras.layers.Conv2D(num_sources * 6, 1, strides=1, padding="same",
                                       activation=None, name="vo_conv8")(conv7)
        poses = tf.keras.layers.GlobalAveragePooling2D("channels_last", name="vo_pred")(poses)
        poses = tf.keras.layers.Reshape((num_sources, 6), name="vo_reshape")(poses)
        posenet = tf.keras.Model(inputs=snippet_image, outputs={"pose": poses}, name="posenet")
        return posenet

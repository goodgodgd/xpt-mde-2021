import tensorflow as tf
from tensorflow.keras import layers

import settings
import model.build_model.model_utils as mu

DISP_SCALING_VGG = 10


class DepthNetBasic:
    """
    Basic DepthNet model used in sfmlearner and geonet
    """
    def __call__(self, input_tensor, input_shape):
        """
        :param input_tensor: input image with size: [batch, height, width, channel]
        :param input_shape: explicit shape (batch, snippet, height, width, channel)
        In the code below, the 'n' in conv'n' or upconv'n' represents scale of the feature map
        conv'n' implies that it is scaled by 1/2^n
        """
        batch, snippet, height, width, channel = input_shape

        conv0 = mu.convolution(input_tensor, 32, 7, strides=1, name="dp_conv0b")
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
        disp3, disp2_up, dpconv3 = self.get_disp_vgg(upconv3, height // 4, width // 4, "dp_disp3")
        upconv2 = self.upconv_with_skip_connection(upconv3, conv2, 64, "dp_up2", disp2_up)  # 1/4
        disp2, disp1_up, dpconv2 = self.get_disp_vgg(upconv2, height // 2, width // 2, "dp_disp2")
        upconv1 = self.upconv_with_skip_connection(upconv2, conv1, 32, "dp_up1", disp1_up)  # 1/2
        disp1, disp0_up, dpconv1 = self.get_disp_vgg(upconv1, height, width, "dp_disp1")
        upconv0 = self.upconv_with_skip_connection(upconv1, disp0_up, 16, "dp_up0")         # 1
        disp0, disp_n1_up, dpconv0 = self.get_disp_vgg(upconv0, height, width, "dp_disp0")

        return [disp0, disp1, disp2, disp3], [dpconv0, dpconv1, dpconv2, dpconv3]

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

    def get_disp_vgg(self, x, dst_height, dst_width, scope):
        conv = layers.Conv2D(1, 3, strides=1, padding="same", activation="sigmoid", name=scope + "_conv")(x)
        disp = layers.Lambda(lambda x: DISP_SCALING_VGG * x + 0.01, name=scope + "_scale")(conv)
        disp_up = mu.resize_image(disp, dst_height, dst_width, scope)
        return disp, disp_up, conv


class DepthNetNoResize(DepthNetBasic):
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

    def get_disp_vgg(self, x, dst_height, dst_width, scope):
        conv = layers.Conv2D(1, 3, strides=1, padding="same", activation="sigmoid", name=scope + "_conv")(x)
        # disp = layers.Lambda(lambda x: (tf.math.exp(x * 15.) + 0.1) * 0.05, name=scope + "_scale")(conv)
        disp = layers.Lambda(lambda x: DISP_SCALING_VGG * x + 0.01, name=scope + "_scale")(conv)
        disp_up = mu.resize_image(disp, dst_height, dst_width, scope)
        return disp, disp_up, conv


class PoseNet:
    def __call__(self, snippet_image, input_shape):
        batch, snippet, height, width, channel = input_shape
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
        return poses

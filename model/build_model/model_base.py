import tensorflow as tf
from tensorflow.keras import layers

import settings
import utils.util_funcs as uf
import model.build_model.model_utils as mu


class ModelBuilderBase:
    def __init__(self, image_shape, batch_size, snippet_len):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.snippet_len = snippet_len

    def get_model(self):
        raise NotImplementedError()

    def build_depth_estim_layers(self, target_image):
        raise NotImplementedError()

    def build_visual_odom_layers(self, image_snippet):
        raise NotImplementedError()


DISP_SCALING_VGG = 10


class BasicModel(ModelBuilderBase):
    """
    Basic VODE model used in sfmlearner and geonet
    """
    def __init__(self, image_shape, batch_size, snippet_len):
        super().__init__(image_shape, batch_size, snippet_len)

    def get_model(self):
        input_img_shape = (self.image_shape[0]*self.snippet_len, self.image_shape[1], self.image_shape[2])
        # input tensor
        stacked_image = layers.Input(shape=input_img_shape, batch_size=self.batch_size, name="image")
        source_image, target_image = layers.Lambda(lambda image: uf.split_into_source_and_target(image),
                                                   name="split_stacked_image")(stacked_image)
        # build a network that outputs depth and pose
        pred_disps_ms = self.build_depth_estim_layers(target_image)
        pred_poses = self.build_visual_odom_layers(stacked_image)
        # create model
        model_input = {"image": stacked_image}
        predictions = {"disp_ms": pred_disps_ms, "pose": pred_poses}
        model = tf.keras.Model(model_input, predictions)
        return model

    def build_depth_estim_layers(self, target_image):
        imheight, imwidth, imchannel = self.image_shape

        conv1 = mu.convolution(target_image, 32, 7, strides=1, name="dp_conv1a")
        conv1 = mu.convolution(conv1, 32, 7, strides=2, name="dp_conv1b")
        conv2 = mu.convolution(conv1, 64, 5, strides=1, name="dp_conv2a")
        conv2 = mu.convolution(conv2, 64, 5, strides=2, name="dp_conv2b")
        conv3 = mu.convolution(conv2, 128, 3, strides=1, name="dp_conv3a")
        conv3 = mu.convolution(conv3, 128, 3, strides=2, name="dp_conv3b")
        conv4 = mu.convolution(conv3, 256, 3, strides=1, name="dp_conv4a")
        conv4 = mu.convolution(conv4, 256, 3, strides=2, name="dp_conv4b")
        conv5 = mu.convolution(conv4, 512, 3, strides=1, name="dp_conv5a")
        conv5 = mu.convolution(conv5, 512, 3, strides=2, name="dp_conv5b")
        conv6 = mu.convolution(conv5, 512, 3, strides=1, name="dp_conv6a")
        conv6 = mu.convolution(conv6, 512, 3, strides=2, name="dp_conv6b")
        conv7 = mu.convolution(conv6, 512, 3, strides=1, name="dp_conv7a")
        conv7 = mu.convolution(conv7, 512, 3, strides=2, name="dp_conv7b")

        upconv7 = self.upconv_with_skip_connection(conv7, conv6, 512, "dp_up7")
        upconv6 = self.upconv_with_skip_connection(upconv7, conv5, 512, "dp_up6")
        upconv5 = self.upconv_with_skip_connection(upconv6, conv4, 256, "dp_up5")
        upconv4 = self.upconv_with_skip_connection(upconv5, conv3, 128, "dp_up4")
        disp4, disp4_up = self.get_disp_vgg(upconv4, int(imheight // 4), int(imwidth // 4), "dp_disp4")
        upconv3 = self.upconv_with_skip_connection(upconv4, conv2, 64, "dp_up3", disp4_up)
        disp3, disp3_up = self.get_disp_vgg(upconv3, int(imheight // 2), int(imwidth // 2), "dp_disp3")
        upconv2 = self.upconv_with_skip_connection(upconv3, conv1, 32, "dp_up2", disp3_up)
        disp2, disp2_up = self.get_disp_vgg(upconv2, imheight, imwidth, "dp_disp2")
        upconv1 = self.upconv_with_skip_connection(upconv2, disp2_up, 16, "dp_up1")
        disp1, disp1_up = self.get_disp_vgg(upconv1, imheight, imwidth, "dp_disp1")

        return [disp1, disp2, disp3, disp4]

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
        disp = layers.Conv2D(1, 3, strides=1, padding="same", activation="sigmoid", name=scope + "_conv")(x)
        disp = layers.Lambda(lambda x: DISP_SCALING_VGG * x + 0.01, name=scope + "_scale")(disp)
        disp_up = mu.resize_image(disp, dst_height, dst_width, scope)
        return disp, disp_up

    def build_visual_odom_layers(self, image_snippet):
        channel_stack_image = layers.Lambda(lambda image: mu.restack_on_channels(image, self.snippet_len),
                                            name="channel_stack")(image_snippet)
        print("[build_visual_odom_layers] channel stacked image shape=", channel_stack_image.get_shape())
        num_sources = self.snippet_len - 1

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


class NoResizingModel(BasicModel):
    """
    Modified BasicModel to remove resizing features in decoding layers
    Width and height of input image must be integer multiple of 128
    """
    def __init__(self, image_shape, batch_size, snippet_len):
        super().__init__(image_shape, batch_size, snippet_len)

    def upconv_with_skip_connection(self, bef_layer, skip_layer, out_channels, scope, bef_pred=None):
        upconv = layers.UpSampling2D(size=(2, 2), interpolation="nearest", name=scope + "_sample")(bef_layer)
        upconv = mu.convolution(upconv, out_channels, 3, strides=1, name=scope + "_conv1")
        upconv = tf.cond(bef_pred is not None,
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer, bef_pred]),
                         lambda: layers.Concatenate(axis=3, name=scope + "_concat")([upconv, skip_layer])
                         )
        upconv = mu.convolution(upconv, out_channels, 3, strides=1, name=scope + "_conv2")
        return upconv

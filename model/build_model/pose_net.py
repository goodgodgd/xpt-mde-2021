import tensorflow as tf
from tensorflow.keras import layers

import model.build_model.model_utils as mu

# TODO: pretrained model (w/ and w/o pretrained weight)이나 conv가 더 많은 모델 등 활용해보기


class PoseNet:
    def __init__(self, input_shape, conv2d):
        self.input_shape = input_shape
        self.conv2d_p = conv2d

    def __call__(self):
        batch, snippet, height, width, channel = self.input_shape
        stacked_image_shape = (snippet*height, width, channel)
        snippet_image = layers.Input(shape=stacked_image_shape, batch_size=batch, name="posenet_input")
        channel_stack_image = layers.Lambda(lambda image: mu.restack_on_channels(image, snippet),
                                            name="channel_stack")(snippet_image)
        print("[PoseNet] channel stacked image shape=", channel_stack_image.get_shape())
        num_sources = snippet - 1

        conv1 = self.conv2d_p(channel_stack_image, 16, 7, strides_=2, name="vo_conv1")
        conv2 = self.conv2d_p(conv1, 32, 5, strides_=2, name="vo_conv2")
        conv3 = self.conv2d_p(conv2, 64, 3, strides_=2, name="vo_conv3")
        conv4 = self.conv2d_p(conv3, 128, 3, strides_=2, name="vo_conv4")
        conv5 = self.conv2d_p(conv4, 256, 3, strides_=2, name="vo_conv5")
        conv6 = self.conv2d_p(conv5, 256, 3, strides_=2, name="vo_conv6")
        conv7 = self.conv2d_p(conv6, 256, 3, strides_=2, name="vo_conv7")

        poses = self.conv2d_p(conv7, num_sources*6, 1, activation_="linear", name="vo_conv8")
        poses = tf.keras.layers.GlobalAveragePooling2D("channels_last", name="vo_pred")(poses)
        poses = tf.keras.layers.Reshape((num_sources, 6), name="vo_reshape")(poses)
        posenet = tf.keras.Model(inputs=snippet_image, outputs={"pose": poses}, name="posenet")
        return posenet


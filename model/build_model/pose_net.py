import tensorflow as tf
from tensorflow.keras import layers

import model.build_model.model_utils as mu


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


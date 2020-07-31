import tensorflow as tf
from tensorflow.keras import layers

# TODO: pretrained model (w/ and w/o pretrained weight)이나 conv가 더 많은 모델 등 활용해보기


class PoseNet:
    def __init__(self, input_shape, global_batch, conv2d):
        self.input_shape = input_shape
        self.global_batch = global_batch
        self.conv2d_p = conv2d
        print("[PoseNet] convolution default options:", input_shape, vars(conv2d))

    def __call__(self):
        batch, snippet, height, width, channel = self.input_shape
        numsrc = snippet - 1
        image_shape = (snippet, height, width, channel)
        input_tensor = layers.Input(shape=image_shape, batch_size=self.global_batch, name="posenet_input")
        # posenet_input: [batch, height, width, snippet*channel]
        posenet_input = layers.Lambda(lambda image: self.restack_on_channels(image),
                                      name="channel_stack")(input_tensor)

        conv1 = self.conv2d_p(posenet_input, 16, 7, strides=2, name="vo_conv1")
        conv2 = self.conv2d_p(conv1, 32, 5, strides=2, name="vo_conv2")
        conv3 = self.conv2d_p(conv2, 64, 3, strides=2, name="vo_conv3")
        conv4 = self.conv2d_p(conv3, 128, 3, strides=2, name="vo_conv4")
        conv5 = self.conv2d_p(conv4, 256, 3, strides=2, name="vo_conv5")
        conv6 = self.conv2d_p(conv5, 256, 3, strides=2, name="vo_conv6")
        conv7 = self.conv2d_p(conv6, 256, 3, strides=2, name="vo_conv7")

        poses = self.conv2d_p(conv7, numsrc*6, 1, activation="linear", name="vo_conv8")
        poses = tf.keras.layers.GlobalAveragePooling2D("channels_last", name="vo_pred")(poses)
        poses = tf.keras.layers.Reshape((numsrc, 6), name="vo_reshape")(poses)
        posenet = tf.keras.Model(inputs=input_tensor, outputs={"pose": poses}, name="posenet")
        return posenet

    def restack_on_channels(self, image5d):
        batch, snippet, height, width, channel = self.input_shape
        # transpose image: [batch, snippet, height, width, channel] -> [batch, height, width, snippet, channel]
        channel_stack_image = tf.transpose(image5d, (0, 2, 3, 1, 4))
        # stack snippet images on channels -> [batch, height, width, snippet*channel]
        channel_stack_image = tf.reshape(channel_stack_image, shape=(batch, height, width, snippet*channel))
        return channel_stack_image

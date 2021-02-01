import tensorflow as tf
from tensorflow.keras import layers
import model.model_util.layer_ops as lo
from model.build_model.pretrained_nets import PretrainedModel
# TODO: pretrained model (w/ and w/o pretrained weight)이나 conv가 더 많은 모델 등 활용해보기


class PoseNetBasic:
    def __init__(self, input_shape, global_batch, conv2d, high_res):
        self.input_shape = input_shape
        self.global_batch = global_batch
        self.conv2d_p = conv2d
        self.high_res = high_res
        print("[PoseNet] convolution default options:", vars(conv2d))

    def __call__(self):
        batch, snippet, height, width, channel = self.input_shape
        numsrc = snippet - 1
        input_tensor, posenet_input = self.create_inputs()

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

    def create_inputs(self):
        batch, snippet, height, width, channel = self.input_shape
        image_shape = (snippet, height, width, channel)
        input_tensor = layers.Input(shape=image_shape, batch_size=self.global_batch, name="posenet_input")
        # posenet_input: [batch, height, width, snippet*channel]
        posenet_input = layers.Lambda(lambda image: self.restack_on_channels(image),
                                      name="channel_stack")(input_tensor)
        return input_tensor, posenet_input


    def restack_on_channels(self, image5d):
        batch, snippet, height, width, channel = self.input_shape
        # transpose image: [batch, snippet, height, width, channel] -> [batch, height, width, snippet, channel]
        channel_stack_image = tf.transpose(image5d, (0, 2, 3, 1, 4))
        # stack snippet images on channels -> [batch, height, width, snippet*channel]
        channel_stack_image = tf.reshape(channel_stack_image, shape=(batch, height, width, snippet*channel))
        return channel_stack_image


class PoseNetImproved(PoseNetBasic):
    def __init__(self, input_shape, global_batch, conv2d, high_res):
        super().__init__(input_shape, global_batch, conv2d, high_res)

    def __call__(self):
        input_tensor, posenet_input = self.create_inputs()

        conv1 = self.conv2d_p(posenet_input, 32, 5, strides=2, name="vo_conv1")
        conv2 = self.conv2d_p(conv1, 32, 5, strides=2, name="vo_conv2")
        conv3 = self.conv2d_p(conv2, 64, 3, strides=2, name="vo_conv3")
        conv4 = self.conv2d_p(conv3, 128, 3, strides=2, name="vo_conv4")
        conv5 = self.conv2d_p(conv4, 256, 3, strides=2, name="vo_conv5")
        conv6 = self.conv2d_p(conv5, 256, 3, strides=2, name="vo_conv6_1")
        conv6 = self.conv2d_p(conv6, 256, 3, name="vo_conv6_2")
        conv6 = self.conv2d_p(conv6, 256, 3, name="vo_conv6_3")

        poses = self.output_process(conv6)
        posenet = tf.keras.Model(inputs=input_tensor, outputs={"pose": poses}, name="posenet")
        return posenet

    def output_process(self, conv6):
        # e.g.      image       conv6
        # kitti     128,512     2, 8
        # kitti_2x  256,1024    4, 16
        batch, snippet, height, width, channel = self.input_shape
        numsrc = snippet - 1

        if self.high_res:
            conv7 = self.conv2d_p(conv6, 512, 3, strides=2, name="vo_conv7_1")
            conv7 = self.conv2d_p(conv7, 512, 3, name="vo_conv7_2")
            conv_last = self.conv2d_p(conv7, 512, 3, name="vo_conv7_3")
        else:
            conv_last = conv6

        print("[PoseNet] output shape before GAP:", conv_last.get_shape())
        poses = self.conv2d_p(conv_last, numsrc*6, 1, activation="linear", name="vo_conv_last")
        poses = tf.keras.layers.GlobalAveragePooling2D("channels_last", name="vo_pred")(poses)
        poses = tf.keras.layers.Reshape((numsrc, 6), name="vo_reshape")(poses)
        return poses


class PoseNetDeep(PoseNetImproved):
    def __init__(self, input_shape, global_batch, conv2d, high_res):
        super().__init__(input_shape, global_batch, conv2d, high_res)

    def __call__(self):
        input_tensor, posenet_input = self.create_inputs()

        conv0 = self.conv2d_p(posenet_input, 32, 5, name="vo_conv0")
        conv1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool1")(conv0)
        conv1 = self.conv2d_p(conv1, 32, 3, name="vo_conv1_1")
        conv1 = self.conv2d_p(conv1, 32, 3, name="vo_conv1_2")

        conv2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool2")(conv1)
        conv2 = self.conv2d_p(conv2, 64, 3, name="vo_conv2_1")
        conv2 = self.conv2d_p(conv2, 32, 1, name="vo_conv2_2")
        conv2 = self.conv2d_p(conv2, 64, 3, name="vo_conv2_3")

        conv3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool3")(conv2)
        conv3 = self.conv2d_p(conv3, 64, 3, name="vo_conv3_1")
        conv3 = self.conv2d_p(conv3, 32, 1, name="vo_conv3_2")
        conv3 = self.conv2d_p(conv3, 64, 3, name="vo_conv3_3")

        conv4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool4")(conv3)
        conv4 = self.conv2d_p(conv4, 128, 3, name="vo_conv4_1")
        conv4 = self.conv2d_p(conv4,  64, 1, name="vo_conv4_2")
        conv4 = self.conv2d_p(conv4, 128, 3, name="vo_conv4_3")

        conv5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool5")(conv4)
        conv5 = self.conv2d_p(conv5, 256, 3, name="vo_conv5_1")
        conv5 = self.conv2d_p(conv5, 128, 1, name="vo_conv5_2")
        conv5 = self.conv2d_p(conv5, 256, 3, name="vo_conv5_3")

        conv6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool6")(conv5)
        conv6 = self.conv2d_p(conv6, 256, 3, name="vo_conv6_1")
        conv6 = self.conv2d_p(conv6, 128, 1, name="vo_conv6_2")
        conv6 = self.conv2d_p(conv6, 256, 3, name="vo_conv6_3")

        poses = self.output_process(conv6)
        posenet = tf.keras.Model(inputs=input_tensor, outputs={"pose": poses}, name="posenet")
        return posenet


class PoseNetPreTrained(PoseNetImproved):
    def __init__(self, input_shape, global_batch, conv2d, high_res, net_name, pretrained):
        super().__init__(input_shape, global_batch, conv2d, high_res)
        self.net_name = net_name
        self.pretrained = pretrained

    def __call__(self):
        input_tensor, posenet_input = self.create_inputs()

        features_ms = PretrainedModel(self.net_name, False).encode(posenet_input)
        conv1, conv2, conv3, conv4, conv5 = features_ms

        conv6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="vo_pool6")(conv5)
        conv6 = self.conv2d_p(conv6, 256, 3, name="vo_conv6_1")
        conv6 = self.conv2d_p(conv6, 128, 1, name="vo_conv6_2")
        conv6 = self.conv2d_p(conv6, 256, 3, name="vo_conv6_3")

        poses = self.output_process(conv6)
        posenet = tf.keras.Model(inputs=input_tensor, outputs={"pose": poses}, name="posenet")
        return posenet


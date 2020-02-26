import tensorflow as tf
from tensorflow.keras import layers


def convolution(x, filters, kernel_size, strides, name):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                  padding="same", activation="relu",
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.025),
                                  # kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  name=name)(x)
    return conv


def resize_like(src, ref, scope):
    ref_height, ref_width = ref.get_shape().as_list()[1:3]
    return resize_image(src, ref_height, ref_width, scope)


def resize_image(src, dst_height, dst_width, scope):
    src_height, src_width = src.get_shape().as_list()[1:3]
    if src_height == dst_height and src_width == dst_width:
        return src
    else:
        return layers.Lambda(lambda image: tf.image.resize(
            image, size=[dst_height, dst_width], method="bilinear"), name=scope+"_resize")(src)


def restack_on_channels(vertical_stack, num_stack):
    batch, imheight, imwidth, _ = vertical_stack.get_shape().as_list()
    imheight = int(imheight // num_stack)
    # create channel for snippet sequence
    channel_stack_image = tf.reshape(vertical_stack, shape=(batch, -1, imheight, imwidth, 3))
    # move snippet dimension to 3
    channel_stack_image = tf.transpose(channel_stack_image, (0, 2, 3, 1, 4))
    # stack snippet images on channels
    channel_stack_image = tf.reshape(channel_stack_image, shape=(batch, imheight, imwidth, -1))
    return channel_stack_image

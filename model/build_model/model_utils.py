import tensorflow as tf
from tensorflow.keras import layers


def conv2d_func_factory(kernel_size=3, strides=1, padding="same", dilation_rate=1,
                        activation="relu", kernel_initializer="glorot_uniform"):

    def conv2d(x, filters, kernel_size_=kernel_size, strides_=strides, padding_=padding,
               dilation_rate_=dilation_rate, activation_=activation,
               kernel_initializer_=kernel_initializer, **kwargs):
        conv = tf.keras.layers.Conv2D(filters, kernel_size_, strides_, padding_,
                                      dilation_rate=dilation_rate_, activation=activation_,
                                      kernel_initializer=kernel_initializer_, **kwargs)(x)
        return conv

    return conv2d


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


# ===== TEST FUNCTIONS

def test_conv2d_factory():
    print("\n===== start test_conv2d_factory")
    conv_op1 = conv2d_func_factory()
    conv_op2 = conv2d_func_factory(kernel_size=5, strides=2, activation="linear")
    print("conv ops are created!")
    x = tf.random.uniform((8, 100, 200, 10), -1, 1)
    y1 = conv_op1(x, 20, name="conv1")
    y2 = conv_op2(x, 30, name="conv2")
    print("conv op1 output shape:", y1.get_shape())
    print("conv op2 output shape:", y2.get_shape())

    print("!!! test_conv2d_factory passed")


if __name__ == "__main__":
    test_conv2d_factory()


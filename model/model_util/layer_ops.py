import tensorflow as tf
from tensorflow.keras import layers


class CustomConv2D:
    def __init__(self, kernel_size=3, strides=1, padding="same", dilation_rate=1,
                 activation="relu", kernel_initializer="glorot_uniform",
                 kernel_regularizer=None, scope=""):
        # set default arguments for Conv2D layer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.scope = scope

    def __call__(self, x, filters, kernel_size=None, strides=None, padding=None, dilation_rate=None,
                 activation=None, kernel_initializer=None, kernel_regularizer=None, name=""):
        # change arguments if there are valid inputs
        kernel_size = self.kernel_size if kernel_size is None else kernel_size
        strides = self.strides if strides is None else strides
        padding = self.padding if padding is None else padding
        dilation_rate = self.dilation_rate if dilation_rate is None else dilation_rate
        activation = self.activation if activation is None else activation
        kernel_initializer = self.kernel_initializer if kernel_initializer is None else kernel_initializer
        kernel_regularizer = self.kernel_regularizer if kernel_regularizer is None else kernel_regularizer
        name = f"{self.scope}_{name}" if self.scope else name

        conv = layers.Conv2D(filters, kernel_size, strides, padding,
                             dilation_rate=dilation_rate, activation=activation,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer,
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


# ===== TEST FUNCTIONS

def test_custom_conv2d():
    print("\n===== start test_conv2d_factory")
    conv_op1 = CustomConv2D()
    conv_op2 = CustomConv2D(kernel_size=5, strides=2, activation="linear", scope="my")
    print("conv ops are created!")
    x = tf.random.uniform((8, 100, 200, 10), -1, 1)
    y1 = conv_op1(x, 20, name="conv1")
    y2 = conv_op2(x, 30, name="conv2")
    print("conv op1 output shape:", y1.get_shape())
    print("conv op2 output shape:", y2.get_shape())
    print("conv op1 args:", vars(conv_op1))
    print("conv op2 args:", vars(conv_op2))
    print("!!! test_conv2d_factory passed")


if __name__ == "__main__":
    test_custom_conv2d()


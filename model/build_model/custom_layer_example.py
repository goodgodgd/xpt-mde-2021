import tensorflow as tf
from tensorflow.keras import layers


class Linear(layers.Layer):
    def __init__(self):
        super(Linear, self).__init__()

    def call(self, x):
        d1 = layers.Dense(6)(x)
        d2 = layers.Dense(8)(d1)
        return d1, d2


def main():
    x = tf.ones((2, 4))
    y1, y2 = Linear()(x)
    z1, z2 = Linear()(y2)
    print("y shape", y1.shape, y2.shape)
    print("z shape", z1.shape, z2.shape)


if __name__ == "__main__":
    main()



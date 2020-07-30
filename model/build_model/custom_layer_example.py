import tensorflow as tf
from tensorflow.keras import layers


class Linear(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        d1 = layers.Dense(6)(x)
        d2 = layers.Dense(8)(d1)
        return d1, d2


class LinearObject(layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = layers.Dense(6)
        self.layer2 = layers.Dense(8)

    def call(self, x):
        d1 = self.layer1(x)
        d2 = self.layer2(d1)
        return d1, d2


def main():
    x = tf.ones((2, 4))
    linear_layer1 = Linear()
    linear_layer2 = LinearObject()
    y1, y2 = linear_layer1(x)
    z1, z2 = linear_layer2(y2)
    print("y shape", y1.shape, y2.shape)
    print("layer1 weights:", len(linear_layer1.weights))
    print("z shape", z1.shape, z2.shape)
    print("layer2 weights:", len(linear_layer2.weights))


if __name__ == "__main__":
    main()



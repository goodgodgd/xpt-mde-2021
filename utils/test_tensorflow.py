import tensorflow as tf
import model.model_util.layer_ops as lo
from tensorflow.keras import layers


class TestTfBug:
    @tf.function
    def test_func_in_class(self):
        print("\n===== start test_func_in_class")
        a = tf.random.uniform((8, 100, 100, 3))
        b = tf.random.uniform((8, 100, 100, 3))
        # !! The backslash(\) cause the problem !!
        c = 0.5 * tf.reduce_mean(tf.abs(a), axis=[1, 2, 3]) + \
            0.5 * tf.reduce_mean(tf.abs(b), axis=[1, 2, 3])
        print("[test_func_in_class]", c)
        return c

    @tf.function
    def test_func_in_class_no_return(self):
        a = tf.random.uniform((8, 100, 100, 3))
        b = tf.random.uniform((8, 100, 100, 3))
        # !! The backslash(\) cause the problem !!
        c = 0.5 * tf.reduce_mean(tf.abs(a), axis=[1, 2, 3]) + 0.5 * tf.reduce_mean(tf.abs(b), axis=[1, 2, 3])


@tf.function
def test_func_just_func():
    print("\n===== start test_func_just_func")
    a = tf.random.uniform((8, 100, 100, 3))
    b = tf.random.uniform((8, 100, 100, 3))
    # !! The backslash(\) here cause the NO problem
    c = 0.5 * tf.reduce_mean(tf.abs(a), axis=[1, 2, 3]) + \
        0.5 * tf.reduce_mean(tf.abs(b), axis=[1, 2, 3])
    print("[test_func_just_func]", c)


def test_model_dict_inout():
    input1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input1")
    conv1 = lo.convolution(input1, 32, 5, strides=1, name="conv1a")
    conv1 = lo.convolution(conv1, 32, 5, strides=2, name="conv1b")
    conv1 = lo.convolution(conv1, 64, 5, strides=1, name="conv1c")

    input2 = layers.Input(shape=(50, 50, 3), batch_size=8, name="input2")
    conv2 = lo.convolution(input2, 32, 5, strides=1, name="conv2a")
    conv2 = lo.convolution(conv2, 32, 5, strides=2, name="conv2b")
    conv2 = lo.convolution(conv2, 64, 5, strides=1, name="conv2c")

    feature = layers.Input(shape=(10, 10), batch_size=8, name="input1")

    inputs = {"input1": input1, "input2": input2}
    outputs = {"output1": conv1, "output2": conv2}
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="double_model")
    model.compile(loss="MSE", optimizer="SGD")
    model.summary()

    tsinput1 = tf.random.uniform(shape=(8, 100, 100, 3), minval=0, maxval=1)
    tsinput2 = tf.random.uniform(shape=(8, 50, 50, 3), minval=0, maxval=1)
    tsfeature = tf.random.uniform(shape=(8, 10, 10), minval=0, maxval=1)

    print("===== dict input: NO problem -> dict output")
    tsinputs = {"input1": tsinput1, "input2": tsinput2}
    predictions = model(tsinputs)
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")

    print("===== list input: NO problem -> dict output")
    predictions = model([tsinput1, tsinput2])
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")

    print("===== dict input with REVERSE order: NO problem -> dict output")
    tsinputs = {"input2": tsinput2, "input1": tsinput1}
    predictions = model(tsinputs)
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")

    print("===== list input with REVERSE order: PROBLEM -> dict output")
    predictions = model([tsinput2, tsinput1])
    for key, pred in predictions.items():
        print(f"predictions: key={key}, shape={pred.get_shape().as_list()}")
    print("first element is used as 'input1', second is used as 'input2'")


def test_name_scope():
    # tf eager execution ignores name scope depending on circumstances
    # behaviour is not consistent
    # below name scope results in error like
    #   ValueError: The name "image" is used 2 times in the model. All layer names should be unique.
    with tf.name_scope("left") as scope:
        image1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="image")
        print("image1 name:", image1.name)  # => image1 name: image:0

    with tf.name_scope("right") as scope:
        image2 = layers.Input(shape=(100, 100, 3), batch_size=8, name="image")
        print("image2 name:", image2.name)  # => image2 name: image_1:0

    output = tf.concat([image1, image2], axis=1)
    try:
        model = tf.keras.Model(inputs=[image1, image2], outputs=output)
        model.summary()
    except Exception as e:
        print("!! Exception:", e)


def test_hierarchy_model():
    input1 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input1")
    conv1 = lo.convolution(input1, 32, 5, strides=1, name="conv1a")
    conv1 = lo.convolution(conv1, 32, 5, strides=2, name="conv1b")
    conv1 = lo.convolution(conv1, 64, 5, strides=1, name="conv1c")
    model1 = tf.keras.Model(inputs=input1, outputs=conv1, name="model1")

    input2 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input2")
    conv2 = lo.convolution(input2, 32, 5, strides=1, name="conv2a")
    conv2 = lo.convolution(conv2, 64, 5, strides=2, name="conv2b")
    conv2 = lo.convolution(conv2, 32, 5, strides=1, name="conv2c")
    model2 = tf.keras.Model(inputs=input2, outputs=conv2, name="model2")

    input3 = layers.Input(shape=(100, 100, 3), batch_size=8, name="input3")
    output1 = model1(input3)
    output2 = model2(input3)
    model = tf.keras.Model(inputs=input3, outputs={"out1": output1, "out2": output2}, name="higher_model")
    model.summary()


if __name__ == "__main__":
    print("tensorflow version:", tf.__version__)

    # This function results in WARNING:tensorflow:AutoGraph could not transform ~~
    TestTfBug().test_func_in_class()

    # This function results in ERROR like
    """
    TypeError: in converted code:
        TypeError: tf__test_func_in_class_no_return() missing 1 required positional argument: 'self'
    """
    # TestTfBug().test_func_in_class_no_return()

    # This function has NOOO problem
    test_func_just_func()

    test_model_dict_inout()

    test_name_scope()

    test_hierarchy_model()

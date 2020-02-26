import tensorflow as tf


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

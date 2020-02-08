import tensorflow as tf


class TestTfBug:
    def test_func_in_class(self):
        a = tf.random.uniform((8, 100, 100, 3))
        b = tf.random.uniform((8, 100, 100, 3))
        # !! The backslash(\) cause the problem !!
        c = 0.5 * tf.reduce_mean(tf.abs(a), axis=[1, 2, 3]) + \
            0.5 * tf.reduce_mean(tf.abs(b), axis=[1, 2, 3])
        return c


def test_func_just_func():
    a = tf.random.uniform((8, 100, 100, 3))
    b = tf.random.uniform((8, 100, 100, 3))
    c = 0.5 * tf.reduce_mean(tf.abs(a), axis=[1, 2, 3]) + \
        0.5 * tf.reduce_mean(tf.abs(b), axis=[1, 2, 3])
    return c


# This function results in WARNING:tensorflow:AutoGraph could not transform ~~
@tf.function
def test_tf_bug_class():
    result = TestTfBug().test_func_in_class()


# This function has no problem
@tf.function
def test_tf_bug_func():
    result = test_func_just_func()


if __name__ == "__main__":
    print("tensorflow version:", tf.__version__)
    test_tf_bug_class()
    test_tf_bug_func()

import sys
import numpy as np
import quaternion
from config import opts
import tensorflow as tf


def print_progress_status(status_msg):
    # Note the \r which means the line should overwrite itself.
    msg = "\r" + status_msg
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def print_numeric_progress(count, total):
    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = f"\r- Progress: {count}/{total}"
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
    if count == total:
        print("")


def input_integer(message, minval=0, maxval=10000):
    while True:
        print(message)
        key = input()
        try:
            key = int(key)
            if key < minval or key > maxval:
                raise ValueError(f"Expected input is within range [{minval}~{maxval}], "
                                 f"but you typed {key}")
        except ValueError as e:
            print(e)
            continue
        break
    return key


def input_float(message, minval=0., maxval=10000.):
    while True:
        print(message)
        key = input()
        try:
            key = float(key)
            if key < minval or key > maxval:
                raise ValueError(f"Expected input is within range [{minval}~{maxval}], "
                                 f"but you typed {key}")
        except ValueError as e:
            print(e)
            continue
        break
    return key


def split_into_source_and_target(stacked_image):
    """
    :param stacked_image: [batch, height*snippet_len, width, 3]
            image sequence is stacked like [im0, im1, im3, im4, im2], im2 is the target
    :return: target_image, [batch, height, width, 3]
             source_image, [batch, height*src_num, width, 3]
    """
    batch, imheight, imwidth, _ = stacked_image.get_shape().as_list()
    imheight = int(imheight // opts.SNIPPET_LEN)
    source_image = tf.slice(stacked_image, (0, 0, 0, 0), (-1, imheight*(opts.SNIPPET_LEN-1), -1, -1))
    target_image = tf.slice(stacked_image, (0, imheight*(opts.SNIPPET_LEN-1), 0, 0),
                            (-1, imheight, -1, -1))
    return source_image, target_image


def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32) * 2 - 1


def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype((im_tensor + 1.) / 2., dtype=tf.uint8)

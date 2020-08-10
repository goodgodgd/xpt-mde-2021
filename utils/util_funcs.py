import sys
import json
import cv2
import os.path as op
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from config import opts


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
    border = imheight*(opts.SNIPPET_LEN-1)
    source_image = stacked_image[:, :border]
    target_image = stacked_image[:, border:]
    return source_image, target_image


def to_float_image(im_tensor):
    return tf.image.convert_image_dtype(im_tensor, dtype=tf.float32) * 2 - 1


def to_uint8_image(im_tensor):
    im_tensor = tf.clip_by_value(im_tensor, -1, 1)
    return tf.image.convert_image_dtype((im_tensor + 1.) / 2., dtype=tf.uint8)


def multi_scale_depths(depth, scales):
    """ shape checked!
    :param depth: [batch, height, width, 1]
    :param scales: list of scales
    :return: list of depths [batch, height/scale, width/scale, 1]
    """
    batch, height, width, _ = depth.get_shape().as_list()
    depth_ms = []
    for sc in scales:
        scaled_size = (int(height // sc), int(width // sc))
        scdepth = tf.image.resize(depth, size=scaled_size, method="nearest")
        depth_ms.append(scdepth)
    return depth_ms


def count_steps(dataset_dir, batch_size=opts.BATCH_SIZE):
    config = read_tfrecords_info(dataset_dir)
    frames = config['length']
    steps = frames // batch_size
    print(f"[count steps] frames={frames}, steps={steps}")
    return steps


def read_tfrecords_info(dataset_dir):
    tfrpath = op.join(opts.DATAPATH_TFR, dataset_dir)
    with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
        config = json.load(fr)
    return config


def check_tfrecord_including(dataset_dir, key_list):
    tfrpath = op.join(opts.DATAPATH_TFR, dataset_dir)
    with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
        config = json.load(fr)

    for key in key_list:
        if key not in config:
            return False
    return True


def read_previous_epoch(model_name):
    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.csv')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            print("[read_previous_epoch] EMPTY history:", history)
            return 0
        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        print("[read_previous_epoch] NO history")
        return 0


def safe_reciprocal_number_ms(src_ms):
    """
    :param src_ms: list of [batch, height/scale, width/scale, 1]
    """
    dst_ms = []
    for i, src in enumerate(src_ms):
        dst = safe_reciprocal_number(src)
        dst_ms.append(dst)
    return dst_ms


def safe_reciprocal_number(src_tensor):
    mask = tf.cast(src_tensor > 0.00001, tf.float32)
    dst_tensor = (1. / src_tensor) * mask
    return dst_tensor


def multi_scale_like_depth(image, depth_ms):
    """
    :param image: [batch, height, width, 3]
    :param depth_ms: list of [batch, height/scale, width/scale, 1]
    :return: image_ms: list of [batch, height/scale, width/scale, 3]
    """
    image_ms = []
    for i, depth in enumerate(depth_ms):
        batch, height_sc, width_sc, _ = depth.get_shape().as_list()
        image_sc = layers.Lambda(lambda img: tf.image.resize(img, size=(height_sc, width_sc), method="bilinear"),
                                 name=f"resize_like_depth_{i}")(image)
        image_ms.append(image_sc)
    return image_ms


def multi_scale_like_flow(image, flow_ms):
    """
    :param image: [batch, height, width, 3]
    :param flow_ms: list of [batch, numsrc, height/scale, width/scale, 1]
    :return: image_ms: list of [batch, height/scale, width/scale, 3]
    """
    image_ms = []
    for i, flow in enumerate(flow_ms):
        batch, numsrc, height_sc, width_sc, _ = flow.get_shape().as_list()
        image_sc = layers.Lambda(lambda img: tf.image.resize(img, size=(height_sc, width_sc), method="bilinear"),
                                 name=f"resize_like_flow_{i}")(image)
        image_ms.append(image_sc)
    return image_ms


def stack_titled_images(view_imgs, guide_lines=True):
    dsize = opts.get_shape("HW")
    location = (20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1
    view = []

    for name, flimage in view_imgs.items():
        flimage_rsz = tf.image.resize(flimage, size=dsize, method="nearest")
        u8image = to_uint8_image(flimage_rsz).numpy()
        if u8image.shape[-1] == 1:
            u8image = cv2.cvtColor(u8image, cv2.COLOR_GRAY2BGR)
        cv2.putText(u8image, name, location, font, font_scale, color, thickness)
        view.append(u8image)

    view = np.concatenate(view, axis=0)
    if guide_lines:
        view[:, 100] = (0, 0, 255)
        view[:, -100] = (0, 0, 255)
    return view


def count_nan(tensor):
    return tf.reduce_sum(tf.cast(tf.math.is_nan(tensor), tf.int32)).numpy()

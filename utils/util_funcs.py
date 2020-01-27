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
    source_image = tf.slice(stacked_image, (0, 0, 0, 0), (-1, imheight*(opts.SNIPPET_LEN-1), -1, -1))
    target_image = tf.slice(stacked_image, (0, imheight*(opts.SNIPPET_LEN-1), 0, 0),
                            (-1, imheight, -1, -1))
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
        scdepth = tf.image.resize(depth, size=scaled_size, method="bilinear")
        depth_ms.append(scdepth)
        # print("[multi_scale_depths] scaled depth shape:", scdepth.get_shape().as_list())
    return depth_ms


def count_steps(dataset_dir, batch_size=opts.BATCH_SIZE):
    tfrpath = op.join(opts.DATAPATH_TFR, dataset_dir)
    with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
        config = json.load(fr)
    frames = config['length']
    steps = frames // batch_size
    print(f"[count steps] frames={frames}, steps={steps}")
    return steps


def check_tfrecord_including(dataset_dir, key_list):
    tfrpath = op.join(opts.DATAPATH_TFR, dataset_dir)
    with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
        config = json.load(fr)

    for key in key_list:
        if key not in config:
            return False
    return True


def read_previous_epoch(model_name):
    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.txt')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            return 0
        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        return 0


def disp_to_depth_tensor(disp_ms):
    target_ms = []
    for i, disp in enumerate(disp_ms):
        target = layers.Lambda(lambda dis: 1./dis, name=f"todepth_{i}")(disp)
        target_ms.append(target)
    return target_ms


def multi_scale_like(image, disp_ms):
    """
    :param image: [batch, height, width, 3]
    :param disp_ms: list of [batch, height/scale, width/scale, 1]
    :return: image_ms: list of [batch, height/scale, width/scale, 3]
    """
    image_ms = []
    for i, disp in enumerate(disp_ms):
        batch, height_sc, width_sc, _ = disp.get_shape().as_list()
        image_sc = layers.Lambda(lambda img: tf.image.resize(img, size=(height_sc, width_sc), method="bilinear"),
                                 name=f"target_resize_{i}")(image)
        image_ms.append(image_sc)
    return image_ms


def make_view(true_target, synth_target, pred_depth, source_image, batidx, srcidx, verbose=True, synth_gt=None):
    dsize = (opts.IM_HEIGHT, opts.IM_WIDTH)
    location = (20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1

    trueim = tf.image.resize(true_target[batidx], size=dsize, method="nearest")
    trueim = to_uint8_image(trueim).numpy()
    cv2.putText(trueim, 'true target image', location, font, font_scale, color, thickness)

    predim = tf.image.resize(synth_target[batidx, srcidx], size=dsize, method="nearest")
    predim = to_uint8_image(predim).numpy()
    cv2.putText(predim, 'reconstructed target image', location, font, font_scale, color, thickness)

    reconim = None
    if synth_gt is not None:
        reconim = tf.image.resize(synth_gt[batidx, srcidx], size=dsize, method="nearest")
        reconim = to_uint8_image(reconim).numpy()
        cv2.putText(reconim, 'reconstructed from gt', location, font, font_scale, color, thickness)

    sourim = to_uint8_image(source_image).numpy()
    sourim = sourim[batidx, opts.IM_HEIGHT * srcidx:opts.IM_HEIGHT * (srcidx + 1)]
    cv2.putText(sourim, 'source image', location, font, font_scale, color, thickness)

    dpthim = tf.image.resize(pred_depth[batidx], size=dsize, method="nearest")
    depth = dpthim.numpy()
    center = (int(dsize[0]/2), int(dsize[1]/2))
    if verbose:
        print("predicted depths\n", depth[center[0]:center[0]+50:10, center[0]-50:center[0]+50:20, 0])
    dpthim = tf.clip_by_value(dpthim, 0., 10.) / 10.
    dpthim = tf.image.convert_image_dtype(dpthim, dtype=tf.uint8).numpy()
    dpthim = cv2.cvtColor(dpthim, cv2.COLOR_GRAY2BGR)
    cv2.putText(dpthim, 'predicted target depth', location, font, font_scale, color, thickness)

    view = [trueim, predim, sourim, dpthim] if reconim is None else [trueim, reconim, predim, sourim, dpthim]
    view = np.concatenate(view, axis=0)
    return view

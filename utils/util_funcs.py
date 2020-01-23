import sys
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


from model.synthesize_batch import synthesize_batch_multi_scale
import model.loss_and_metric as lm
import cv2
import numpy as np


def make_reconstructed_views(model, dataset):
    recon_views = []
    for i, features in enumerate(dataset):
        predictions = model(features['image'])
        pred_disp_ms = predictions['disp_ms']
        pred_pose = predictions['pose']
        pred_depth_ms = lm.disp_to_depth(pred_disp_ms)
        print("predicted snippet poses:\n", pred_pose[0].numpy())

        # reconstruct target image
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        source_image, target_image = split_into_source_and_target(stacked_image)
        true_target_ms = lm.multi_scale_like(target_image, pred_disp_ms)
        synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, pred_depth_ms, pred_pose)

        # make stacked image of [true target, reconstructed target, source image, predicted depth]
        #   in 1/1 scale
        view1 = extract_view(true_target_ms, synth_target_ms, pred_depth_ms, source_image, sclidx=0, batidx=0, srcidx=0)
        #   in 1/4 scale
        # view2 = extract_view(true_target_ms, synth_target_ms, pred_depth_ms, source_image, sclidx=2, batidx=0, srcidx=0)
        # view = np.concatenate([view1, view2], axis=1)
        recon_views.append(view1)
        if i >= 10:
            break

    return recon_views


def extract_view(true_target_ms, synth_target_ms, pred_depth_ms, source_image, sclidx, batidx, srcidx):
    dsize = (opts.IM_HEIGHT, opts.IM_WIDTH)
    location = (20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1

    trueim = tf.image.resize(true_target_ms[sclidx][batidx], size=dsize, method="nearest")
    trueim = to_uint8_image(trueim).numpy()
    cv2.putText(trueim, 'true target image', location, font, font_scale, color, thickness)

    predim = tf.image.resize(synth_target_ms[sclidx][batidx, srcidx], size=dsize, method="nearest")
    predim = to_uint8_image(predim).numpy()
    cv2.putText(predim, 'reconstructed target image', location, font, font_scale, color, thickness)

    sourim = to_uint8_image(source_image).numpy()
    sourim = sourim[batidx, opts.IM_HEIGHT * srcidx:opts.IM_HEIGHT * (srcidx + 1)]
    cv2.putText(sourim, 'source image', location, font, font_scale, color, thickness)

    dpthim = tf.image.resize(pred_depth_ms[sclidx][batidx], size=dsize, method="nearest")
    depth = dpthim.numpy()
    center = (int(dsize[0]/2), int(dsize[1]/2))
    print("predicted depths\n", depth[center[0]:center[0]+50:10, center[0]-50:center[0]+50:20, 0])
    dpthim = tf.clip_by_value(dpthim, 0., 10.) / 10.
    dpthim = tf.image.convert_image_dtype(dpthim, dtype=tf.uint8).numpy()
    dpthim = cv2.cvtColor(dpthim, cv2.COLOR_GRAY2BGR)
    cv2.putText(dpthim, 'predicted target depth', location, font, font_scale, color, thickness)

    view = np.concatenate([trueim, predim, sourim, dpthim], axis=0)
    return view

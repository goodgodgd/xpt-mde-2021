import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as util
from model.synthesize_batch import synthesize_batch_view


def photometric_loss_multi_scale(synthesized_target_ms, original_target_ms):
    """
    :param synthesized_target_ms: multi scale synthesized targets, list of
                                  [batch, num_src, height/scale, width/scale, 3]
    :param original_target_ms: multi scale target images, list of [batch, height, width, 3]
    :return: photo_loss scalar
    """
    losses = []
    for i, (synt_target, orig_target) in enumerate(zip(synthesized_target_ms, original_target_ms)):
        loss = layers.Lambda(lambda inputs: photometric_loss(inputs), name=f"photo_loss_{i}")\
                            ([synt_target, orig_target])
        losses.append(loss)
    photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=0)),
                               name="photo_loss_sum")(losses)
    return photo_loss


def photometric_loss(inputs):
    """
    :param inputs:
        synt_target: scaled synthesized target image [[batch, num_src, height/scale, width/scale, 3]]
        target_image: scaled original target image [batch, height/scale, width/scale, 3]
    :return: scalar loss
    """
    synt_target, orig_target = inputs
    orig_target = tf.expand_dims(orig_target, axis=1)
    # create mask to ignore black region
    synt_target_gray = tf.reduce_mean(synt_target, axis=-1, keepdims=True)
    error_mask = tf.equal(synt_target_gray, 0)

    # target_image_sc [batch, 1, height/scale, width/scale, 3]
    # axis=1 broadcasted in subtraction
    photo_error = tf.abs(synt_target - orig_target)
    photo_error = tf.where(error_mask, tf.constant(0, dtype=tf.float32), photo_error)
    photo_loss = tf.reduce_mean(photo_error)
    return photo_loss


def smootheness_loss_multi_scale(disp_ms, image_ms):
    """
    :param disp_ms: multi scale disparity map, list of [batch, height/scale, width/scale, 1]
    :param image_ms: multi scale image, list of [batch, height/scale, width/scale, 3]
    :return:
    """
    losses = []
    for i, (disp, image) in enumerate(zip(disp_ms, image_ms)):
        loss = layers.Lambda(lambda inputs: smootheness_loss(inputs),
                             name=f"smooth_loss_{i}")([disp, image])
        losses.append(loss)
    photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=0)),
                               name="smooth_loss_sum")(losses)
    return photo_loss


def smootheness_loss(inputs):
    """
    :param inputs:
        disp: scaled disparity map, list of [batch, height/scale, width/scale, 1]
        image: scaled original target image [batch, height/scale, width/scale, 3]
    :return: scalar loss
    """
    disp, image = inputs

    def gradient_x(img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(image)
    image_gradients_y = gradient_y(image)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))



import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as util
from model.synthesize_batch import synthesize_batch_view


def photometric_loss_multi_scale(synthesized_target_ms, target_image):
    """
    :param synthesized_target_ms: list of multi scale synthesized targets
                                [[batch, num_src, height/scale, width/scale, 3]]
    :param target_image: [batch, height, width, 3]
    :return: photo_loss scalar
    """
    losses = []
    for i, synt_target in enumerate(synthesized_target_ms):
        loss = layers.Lambda(lambda inputs: photometric_loss(inputs), name=f"photo_loss_{i}")\
                            ([synt_target, target_image])
        losses.append(loss)
    photo_loss = layers.Lambda(lambda data: tf.reduce_sum(tf.stack(data, axis=0)),
                               name="photo_loss_sum")(losses)
    return photo_loss


def photometric_loss(inputs):
    """
    :param inputs:
        synt_target: synthesized target image that was scaled [[batch, num_src, height/scale, width/scale, 3]]
        target_image: original target image [batch, height, width, 3]
    :return: scalar loss
    """
    synt_target, target_image = inputs
    batch, num_src, height_sc, width_sc, _ = synt_target.get_shape().as_list()
    target_image_sc = tf.image.resize(target_image, size=(height_sc, width_sc), method="bilinear")
    target_image_sc = tf.expand_dims(target_image_sc, axis=1)
    synt_target_gray = tf.reduce_mean(synt_target, axis=-1, keepdims=True)
    error_mask = tf.equal(synt_target_gray, 0)
    photo_error = tf.abs(synt_target - target_image_sc)
    photo_error = tf.where(error_mask, tf.constant(0, dtype=tf.float32), photo_error)
    photo_loss = tf.reduce_mean(photo_error)
    return photo_loss



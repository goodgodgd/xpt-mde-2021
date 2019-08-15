import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as util
from model.synthesize_batch import synthesize_batch_view


def synthesize_view_multi_scale(stacked_image, intrinsic, pred_depth_ms, pred_pose):
    """
    :param stacked_image: [batch, height*5, width, 3]
    :param intrinsic: [batch, 3, 3]
    :param pred_depth_ms: predicted depth in multi scale [batch, height*scale, width*scale, 1]
    :param pred_pose: predicted source pose [batch, num_src, 4, 4]
    :return: reconstructed target view
    """
    width_ori = stacked_image.get_shape().as_list()[2]
    # convert pose vector to transformation matrix
    poses_matr = util.pose_rvec2matr_batch(pred_pose)
    recon_images = []
    for key, depth_sc in pred_depth_ms.items():
        batch, height_sc, width_sc, _ = pred_depth_ms[0].get_shape().as_list()
        scale = int(width_ori // width_sc)
        # adjust intrinsic upto scale
        intrinsic_sc = scale_intrinsic(intrinsic, scale)
        # reorganize source images: [batch, 4, height, width, 3]
        source_images_sc = layers.Lambda(lambda image: reshape_source_images(image, scale),
                                         name="reorder_source")(stacked_image)
        print("[synthesize_view_multi_scale] source image shape=", source_images_sc.get_shape())
        recon_image_sc = synthesize_batch_view(source_images_sc, depth_sc, poses_matr, intrinsic_sc)

    return recon_images


def scale_intrinsic(intrinsic, scale):
    batch = intrinsic.get_shape().as_list()[0]
    scaled_part = intrinsic[:, :2, :] / scale
    const_part = tf.tile(tf.constant([[[0, 0, 1]]], dtype=tf.float32), (batch, 1, 1))
    scaled_intrinsic = tf.concat([scaled_part, const_part], axis=1)
    return scaled_intrinsic


def reshape_source_images(stacked_image, scale):
    """
    :param stacked_image: [batch, 5*height, width, 3]
    :param scale: scale to reduce image size
    :return: reorganized source images [batch, 4, height, width, 3]
    """
    # resize image
    batch, stheight, stwidth, _ = stacked_image.get_shape().as_list()
    scaled_size = (int(stheight//scale), int(stwidth//scale))
    scaled_image = tf.image.resize(stacked_image, size=scaled_size, method="bilinear")
    # slice only source images
    batch, scheight, scwidth, _ = scaled_image.get_shape().as_list()
    scheight = int(scheight // opts.SNIPPET_LEN)
    source_images = tf.slice(scaled_image, (0, 0, 0, 0), (-1, scheight*(opts.SNIPPET_LEN - 1), -1, -1))
    # reorganize source images: (4*height,) -> (4, height)
    source_images = tf.reshape(source_images, shape=(batch, -1, scheight, scwidth, 3))
    return source_images

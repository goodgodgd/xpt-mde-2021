import os.path as op
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import settings
from config import opts
import utils.util_funcs as uf


def synthesize_view_multi_scale(stacked_image, intrinsic, pred_depth_ms, pred_pose):
    """
    :param stacked_image: [batch, height*5, width, 3]
    :param intrinsic: [batch, 3, 3]
    :param pred_depth_ms: predicted depth in multi scale [batch, height*scale, width*scale, 1]
    :param pred_pose: predicted source pose [batch, 4, 6]
    :return: reconstructed target view
    """
    width_ori = stacked_image.get_shape().as_list()[2]
    # convert pose vector to transformation matrix
    poses_matr = uf.pose_rvec2matr_batch(pred_pose)
    recon_images = []
    for depth_sc in pred_depth_ms.items():
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


def synthesize_batch_view(src_image, tgt_depth, pose, intrinsic):
    """
    :param src_image: source image nearby the target image [batch, num_src, height, width, 3]
    :param tgt_depth: depth map of the target image in meter scale [batch, height, width, 1]
    :param pose: pose matrices that transform points from target to source frame [batch, num_src, 4, 4]
    :param intrinsic: camera projection matrix [batch, 3, 3]
    :return: synthesized target image [batch, num_src, height, width, 3]
    """

    batch, num_src, height, width, chann = src_image.get_shape().as_list()
    # debug_coords = pixel_meshgrid(height, width, 80)
    # debug_inds = tf.cast(debug_coords[0, 16:-16] + debug_coords[1, 16:-16]*width, tf.int32)
    tgt_pixel_coords = pixel_meshgrid(height, width)
    # print("pixel coordintes", tf.gather(tgt_pixel_coords[:2, :], debug_inds, axis=1))

    tgt_cam_coords = pixel2cam(tgt_pixel_coords, tgt_depth, intrinsic)
    src_cam_coords = transform_to_source(tgt_cam_coords, pose)
    # debug_cam_coords = tf.transpose(tf.concat((tgt_cam_coords[:3, :], src_cam_coords[:3, :]), axis=0))
    # print("tgt src points compare", tf.gather(debug_cam_coords, debug_inds))

    src_pixel_coords = cam2pixel(src_cam_coords, intrinsic)
    # print("src_pixel_coords", tf.gather(src_pixel_coords[:2, :], debug_inds, axis=1))

    tgt_image_synthesized = reconstruct_bilinear_interp(src_pixel_coords, src_image, tgt_depth)
    print("reconstructed image", tgt_image_synthesized.get_shape(), tgt_image_synthesized.dtype)
    tgt_image_synthesized = tgt_image_synthesized.numpy()
    return tgt_image_synthesized


def pixel_meshgrid(height, width, stride=1):
    """
    :return: pixel coordinates like vectors of (u,v,1) [3, height*width]
    """
    v = np.linspace(0, height-stride, int(height//stride))
    u = np.linspace(0, width-stride,  int(width//stride))
    ugrid, vgrid = tf.meshgrid(u, v)
    uv = tf.stack([ugrid, vgrid], axis=0)
    uv = tf.reshape(uv, (2, -1))
    num_pts = uv.get_shape().as_list()[1]
    uv = tf.concat([uv, tf.ones((1, num_pts), tf.float64)], axis=0)
    return uv


def pixel2cam(pixel_coords, depth, intrinsic):
    """
    :param pixel_coords: (u,v,1) [3, height*width]
    :param depth: [batch, height, width]
    :param intrinsic: [batch, 3, 3]
    :return: 3D points like (x,y,z,1) in target frame [batch, 4, height*width]
    """
    batch = depth.get_shape().as_list()[0]
    depth = tf.tile(tf.reshape(depth, (batch, 1, -1)), (1, 3, 1))
    # calc sum of products over specified dimension
    # cam_coords[i, j, k] = inv(intrinsic)[i, j, :] dot pixel_coords[:, k]
    # [batch, 3, height*width] = [batch, 3, 3] x [3, height*width]
    cam_coords = tf.tensordot(tf.linalg.inv(intrinsic), pixel_coords, [[2], [0]])
    # [batch, 3, height*width] = [batch, 3, height*width] * [batch, 3, height*width]
    cam_coords *= depth
    # num_pts = height * width
    num_pts = cam_coords.get_shape().as_list()[2]
    # make homogeneous coordinates
    cam_coords = tf.concat([cam_coords, tf.ones((batch, 1, num_pts), tf.float32)], axis=1)
    return cam_coords


def transform_to_source(tgt_coords, t2s_pose):
    """
    :param tgt_coords: target frame coordinates like (x,y,z,1) [batch, 4, height*width]
    :param t2s_pose: pose matrices that transform points from target to source frame [batch, num_src, 4, 4]
    :return: transformed points in source frame like (x,y,z,1) [batch, num_src, 4, height*width]
    """
    num_src = t2s_pose.get_shape().as_list()[1]
    tgt_coords_expand = tf.expand_dims(tgt_coords, 1)
    tgt_coords_expand = tf.tile(tgt_coords_expand, (1, num_src, 1, 1))
    # [batch, num_src, 4, height*width] = [batch, num_src, 4, 4] x [batch, num_src, 4, height*width]
    src_coords = tf.matmul(t2s_pose, tgt_coords_expand)
    return src_coords


def cam2pixel(cam_coords, intrinsic):
    """
    :param cam_coords: 3D points in source frame (x,y,z,1) [batch, num_src, 4, height*width]
    :param intrinsic: intrinsic camera matrix [batch, 3, 3]
    :return: projected pixel coordinates on source image plane (u,v,1) [batch, num_src, 3, height*width]
    """
    num_src = cam_coords.get_shape().as_list()[1]
    intrinsic_expand = tf.expand_dims(intrinsic, 1)
    # [batch, num_src, 3, 3]
    intrinsic_expand = tf.tile(intrinsic_expand, (1, num_src, 1, 1))
    # [batch, num_src, 3, height*width] = [batch, num_src, 3, 3] x [batch, num_src, 3, height*width]
    pixel_coords = tf.matmul(intrinsic_expand, cam_coords[:, :, :3, :])
    # normalize scale
    pixel_scales = pixel_coords[:, :, 2, :]
    pixel_scales = tf.tile(pixel_scales, (1, 1, 3, 1))
    pixel_coords = pixel_coords / (pixel_scales + 1e-10)
    return pixel_coords


def reconstruct_bilinear_interp(pixel_coords, image, depth):
    """
    :param pixel_coords: floating-point pixel coordinates (u,v,1) [batch, num_src, 3, height*width]
    :param image: source image [batch, num_src, height, width, 3]
    :param depth: target depth image [batch, height, width, 1]
    :return: reconstructed image [batch, num_src, height, width, 3]
    """
    # pad 1 pixel around image
    top_pad, bottom_pad, left_pad, right_pad = (1, 1, 1, 1)
    paddings = tf.constant([[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]])
    padded_image = tf.pad(image, paddings, "CONSTANT")
    print(padded_image[0:5, 0:5, 0])
    # adjust pixel coordinates for padded image
    pixel_coords = tf.round(pixel_coords[:2, :] + 1)
    # clip pixel coordinates into padded image region as integer
    ih, iw, ic = image.get_shape().as_list()
    u_coords = tf.clip_by_value(pixel_coords[0, :], 0, iw+1)
    v_coords = tf.clip_by_value(pixel_coords[1, :], 0, ih+1)
    # pixel as (v, u) in rows for gather_nd()
    pixel_coords = tf.stack([v_coords, u_coords], axis=1)
    pixel_coords = tf.cast(pixel_coords, tf.int32)

    # sample pixels from padded image
    flat_image = tf.gather_nd(padded_image, pixel_coords)
    # set black in depth-zero region
    depth_vec = tf.reshape(depth, shape=(-1, 1))
    depth_invalid_mask = tf.math.equal(depth_vec, 0)
    zeros = tf.zeros(flat_image.get_shape(), dtype=tf.uint8)
    flat_image = tf.where(depth_invalid_mask, zeros, flat_image)
    recon_image = tf.reshape(flat_image, shape=(ih, iw, ic))

    return recon_image


def vode_loss(y_true, y_pred):
    # print(y_true.keys())
    print(y_true.get_shape().as_list())
    print(y_pred.get_shape().as_list())
    loss = tf.keras.backend.mean(y_pred, axis=None)
    # photometric_loss = calc_photometric_loss(y_true, y_pred)
    return loss


# ==================== tests ====================
def test_reshape_source_images():
    print("===== start test_reshape_source_images")
    filename = op.join(opts.DATAPATH_SRC, "kitti_raw_train", "2011_09_26_0001", "000024.png")
    image = cv2.imread(filename)
    batch_image = np.expand_dims(image, 0)
    batch_image = np.tile(batch_image, (8, 1, 1, 1))
    print("batch image shape", batch_image.shape)
    batch_image_tensor = tf.constant(batch_image, dtype=tf.float32)

    sources = reshape_source_images(batch_image_tensor, 2)

    sources = tf.cast(sources, tf.uint8)
    sources = sources.numpy()
    print("reordered source image shape", sources.shape)
    # cv2.imshow("original image", image)
    # cv2.imshow("reordered image1", sources[0, 1])
    # cv2.imshow("reordered image2", sources[0, 2])
    # cv2.waitKey()
    # assert (image[opts.IM_HEIGHT:opts.IM_HEIGHT*2] == sources[0, 1]).all()
    print("!!! test_reshape_source_images passed")


def test_scale_intrinsic():
    intrinsic = np.array([8, 0, 4, 0, 8, 4, 0, 0, 1], dtype=np.float32).reshape((1, 3, 3))
    intrinsic = tf.constant(np.tile(intrinsic, (8, 1, 1)))
    scale = 2

    intrinsic_sc = scale_intrinsic(intrinsic, scale)

    print("scaled intrinsic:", intrinsic_sc[0])
    assert np.isclose((intrinsic[:, :2, :]/2), intrinsic_sc[:, :2, :]).all()
    assert np.isclose((intrinsic[:, -1, :]), intrinsic_sc[:, -1, :]).all()
    print("!!! test_scale_intrinsic passed")


def test_pixel2cam():
    batch, height, width = (8, 4, 4)
    tgt_pixel_coords = pixel_meshgrid(height, width)
    tgt_pixel_coords = tf.cast(tgt_pixel_coords, dtype=tf.float32)
    intrinsic = np.array([4, 0, height/2, 0, 4, width/2, 0, 0, 1], dtype=np.float32).reshape((1, 3, 3))
    intrinsic = tf.constant(np.tile(intrinsic, (batch, 1, 1)), dtype=tf.float32)
    depth = tf.ones((batch, height, width), dtype=tf.float32) * 2

    tgt_cam_coords = pixel2cam(tgt_pixel_coords, depth, intrinsic)

    print(tgt_cam_coords[0])
    assert (tgt_cam_coords.get_shape() == (batch, 4, height*width))
    print("!!! test_pixel2cam passed")


def test_transform_to_source():
    batch, num_pts, num_src = (8, 6, 3)
    coords = np.arange(1, 4*num_pts+1).reshape((num_pts, 4)).T
    coords[3, :] = 1
    coords = np.tile(coords, (batch, 1, 1))
    print(f"coordinates: {coords.shape}\n{coords[2]}")

    poses = np.identity(4)*2
    poses[3, 3] = 1
    poses = np.tile(poses, (batch, num_src, 1, 1))
    print(f"poses: {poses.shape}\n{poses[2, 1]}")

    coords = tf.constant(coords, dtype=tf.float32)
    poses = tf.constant(poses, dtype=tf.float32)
    src_coords = transform_to_source(coords, poses)
    print(f"src coordinates: {src_coords.get_shape()}\n{src_coords[2, 1]}")

    assert np.isclose(coords[2, :3]*2, src_coords[2, 1, :3]).all()
    print("!!! test_transform_to_source passed")


def test():
    test_reshape_source_images()
    test_scale_intrinsic()
    test_pixel2cam()
    test_transform_to_source()


if __name__ == "__main__":
    test()

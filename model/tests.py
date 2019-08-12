import os
import tensorflow as tf
import numpy as np
import quaternion

import settings
import model.synthesize_single_view as sv


def test_linspace():
    # tf ops must take float variables
    # better use np.linspace instead
    x = tf.linspace(0., 3., 4)
    print("test_linspace passed")


def test_pixel_meshgrid():
    height, width = (3, 4)
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    XY = sv.pixel_meshgrid(height, width)
    # print("meshgrid result", XY)
    out_height, out_width = XY.get_shape().as_list()
    assert (out_height == 3), f"out_width={out_height}"
    assert (out_width == height*width)
    assert (np.isclose(XY[0, :width].numpy(), x).all())
    assert (np.isclose(XY[0, -width:].numpy(), x).all())
    assert (np.isclose(XY[1, :height].numpy(), y[0]).all())
    assert (np.isclose(XY[1, -height:].numpy(), y[-1]).all())
    print("test_pixel_meshgrid passed")


def test_gather():
    coords = tf.tile(tf.expand_dims(tf.linspace(0., 10., 11), 1), (1, 3))
    # print(coords)
    indices = tf.cast(tf.linspace(0., 10., 6), tf.int32)
    extracted = tf.gather(coords, indices)
    # print(extracted)
    assert (np.isclose(extracted[:, 0].numpy(), indices.numpy()).all())
    print("test_gather passed")


def test_pixel2cam2pixel():
    height, width = (5, 7)
    intrinsic = tf.constant([[5, 0, width//2], [0, 5, height//2], [0, 0, 1]], dtype=tf.float64)
    depth = tf.ones(shape=(height, width), dtype=tf.float64)*3
    pixel_coords = sv.pixel_meshgrid(height, width)
    cam_coords = sv.pixel2cam(pixel_coords, depth, intrinsic)
    prj_pixel_coords = sv.cam2pixel(cam_coords, intrinsic)
    # print("projected pixel coordinates\n", prj_pixel_coords.numpy())
    assert (np.isclose(pixel_coords.numpy(), prj_pixel_coords.numpy()).all())
    print("test_pixel2cam2pixel passed")


def test_pad():
    img = tf.ones((4, 5, 3), dtype=tf.float32)
    # print("original channel 0", img[:, :, 0])
    paddings = tf.constant([[1, 1], [1, 1], [0, 0]])
    pad = tf.pad(img, paddings, "CONSTANT")
    # print("paddings", paddings)
    # print("padded shape:", pad.shape)
    print("padded channel 0", pad[:, :, 1])


def test_gather_nd():
    mesh = sv.pixel_meshgrid(3, 4)
    mesh = tf.reshape(tf.transpose(mesh[:2, :]), (3, 4, -1))
    # print("mesh", mesh)
    indices = tf.constant([[0, 0], [1, 2]])
    expect = tf.stack((mesh[0, 0, :], mesh[1, 2, :]), axis=0)
    result = tf.gather_nd(mesh, indices)
    # print("gather_nd expect", expect)
    # print("gather_nd result", result)
    assert (np.isclose(expect, result).all())
    print("test_gather_nd passed")


def test_rotation_vector():
    quat = quaternion.from_float_array(np.array([np.cos(np.pi/3), 0, np.sin(np.pi/3), 0]))
    print("quaterion angle pi*2/3 about y-axis", quat)
    rvec = quaternion.as_rotation_vector(quat)
    print("rotation vector:", rvec)
    assert (np.isclose(np.linalg.norm(rvec), np.pi*2/3))
    print("test_rotation_vector passed")


def test():
    np.set_printoptions(precision=3, suppress=True)
    test_linspace()
    test_pixel_meshgrid()
    test_gather()
    test_pixel2cam2pixel()
    test_pad()
    test_gather_nd()
    test_rotation_vector()


if __name__ == "__main__":
    test()

import os
import tensorflow as tf
import numpy as np
import quaternion
import datetime
import time


def test_linspace():
    # tf ops must take float variables
    # better use np.linspace instead
    x = tf.linspace(0., 3., 4)
    print("linspace", x)


def test_gather():
    coords = tf.tile(tf.expand_dims(tf.linspace(0., 10., 11), 1), (1, 3))
    # print(coords)
    indices = tf.cast(tf.linspace(0., 10., 6), tf.int32)
    extracted = tf.gather(coords, indices)
    # print(extracted)
    assert (np.isclose(extracted[:, 0].numpy(), indices.numpy()).all())
    print("!!! test_gather passed")


def test_pad():
    img = tf.ones((4, 5, 3), dtype=tf.float32)
    # print("original channel 0", img[:, :, 0])
    paddings = tf.constant([[1, 1], [1, 1], [0, 0]])
    pad = tf.pad(img, paddings, "CONSTANT")
    # print("paddings", paddings)
    # print("padded shape:", pad.shape)
    print("padded channel 0", pad[:, :, 1])


def test_rotation_vector():
    quat = quaternion.from_float_array(np.array([np.cos(np.pi/3), 0, np.sin(np.pi/3), 0]))
    print("quaterion angle pi*2/3 about y-axis", quat)
    rvec = quaternion.as_rotation_vector(quat)
    print("rotation vector:", rvec)
    assert (np.isclose(np.linalg.norm(rvec), np.pi*2/3))
    print("!!! test_rotation_vector passed")


def test_time():
    nowtime = datetime.datetime.now()
    print("nowtime", nowtime)
    print("formatted time", nowtime.strftime("%m%d_%H%M%S"))
    print("asctime", time.asctime())


def test_casting():
    data = "1.1"
    try:
        data = int(data)
    except Exception as e:
        print(e)
        print(type(e))
        print(str(e))


def test_rotation_conversion():
    print("\n[test_rotation_conversion]")
    angle = np.pi/3
    # fixed axis
    axis = np.array([0, 0, 1.])
    rvec = axis * angle
    # rotation matrix for the pose rotated by 'angle' along z-axis
    rotm = np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    print(rotm)
    # NEGATIVE skew symmetric matrix for cross product with 'axis'(=normalized rvec)
    # TODO: NOTE! pose를 표현할 때는 skew symmetric matrix의 부호를 반대로 해야 한다.
    what = np.array([[0, axis[2], -axis[1]], [-axis[2], 0, axis[0]], [axis[1], -axis[0], 0]])
    rotm_conv = np.eye(3) + what * np.sin(angle) + np.dot(what, what) * (1 - np.cos(angle))
    print(rotm_conv)
    assert np.isclose(rotm, rotm_conv).all()


def test():
    np.set_printoptions(precision=3, suppress=True)
    test_linspace()
    test_gather()
    test_pad()
    test_rotation_vector()
    test_time()
    test_casting()
    test_rotation_conversion()


if __name__ == "__main__":
    test()

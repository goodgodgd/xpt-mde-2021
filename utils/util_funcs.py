import sys
import numpy as np
import quaternion
import tensorflow as tf


def print_progress(count, is_total: bool = False):
    if is_total:
        # static variable in function
        print_progress.total = getattr(print_progress, 'last_hour', count)
    else:
        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {}/{}".format(count, print_progress.total)
        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    if count == print_progress.total:
        print("")


def pose_quat2mat(pose):
    t = np.expand_dims(pose[:3], axis=1)
    q = pose[3:]
    norm = np.linalg.norm(q)
    q = q / norm
    q = quaternion.from_float_array(q)
    # transpose: this rotation matrix is for "frame" rotation, not point rotation
    rot = quaternion.as_rotation_matrix(q).T
    mat = np.concatenate([rot, t], axis=1)
    mat = np.concatenate([mat, np.array([[0, 0, 0, 1]])], axis=0)
    return mat


def pose_mat2quat(pose):
    trans = pose[:3, 3]
    rot_mat = pose[:3, :3].T
    quat = quaternion.from_rotation_matrix(rot_mat)
    quat = quaternion.as_float_array(quat)
    quat = quat / np.linalg.norm(quat)
    pose_quat = np.concatenate([trans, quat], axis=0)
    return pose_quat


def pose_rvec2matr_batch(poses):
    """
    :param poses: poses with twist coordinates, (tx, ty, tz, u1, u2, u3) [batch, N, 6]
    :return: poses in transformation matrix [batch, N, 4, 4]
    """
    poses = tf.expand_dims(poses, -1)
    batch, snippet, _, _ = poses.get_shape().as_list()
    trans = poses[:, :, :3, :]
    uvec = poses[:, :, 3:, :]
    unorm = tf.expand_dims(tf.linalg.norm(uvec, axis=2), axis=2)
    uvec = uvec / unorm
    w1 = uvec[:, :, 0:1, :]
    w2 = uvec[:, :, 1:2, :]
    w3 = uvec[:, :, 2:3, :]
    z = tf.zeros(shape=(batch, snippet, 1, 1))
    w_hat = tf.concat([z, -w3, w2, w3, z, -w1, -w2, w1, z], axis=2)
    w_hat = tf.reshape(w_hat, shape=(batch, snippet, 3, 3))
    identity = tf.expand_dims(tf.expand_dims(tf.eye(3), axis=0), axis=0)
    identity = tf.tile(identity, (batch, snippet, 1, 1))
    rotmat = identity + w_hat*tf.sin(unorm) + tf.matmul(w_hat, w_hat)*(1 - tf.cos(unorm))

    tmat = tf.concat([rotmat, trans], axis=3)
    last_row = tf.tile(tf.constant([[[[0, 0, 0, 1]]]], dtype=tf.float32), multiples=(batch, snippet, 1, 1))
    tmat = tf.concat([tmat, last_row], axis=2)
    return tmat


# ==================== tests ====================

def test_pose_quat2mat():
    rmat = pose_quat2mat(np.array([0, 0, 0, 1, 0, 0, 0]))
    assert (np.isclose(np.identity(4), rmat).all()), \
        f"[test_pose_quat2mat], np.identity(4) != \n{rmat}"
    print("test_pose_quat2mat passed")


def test_convert_pose():
    pose_quat1 = np.array([1, 2, 3, np.cos(np.pi/3), 0, 0, np.sin(np.pi/3)])
    pose_mat = pose_quat2mat(pose_quat1)
    pose_quat2 = pose_mat2quat(pose_mat)

    quat1 = quaternion.from_float_array(pose_quat1[3:])
    quat2 = quaternion.from_float_array(pose_quat2[3:])
    rotm1 = quaternion.as_rotation_matrix(quat1)
    rotm2 = quaternion.as_rotation_matrix(quat2)
    print(f"convert pose quat1={quat1}, quat2={quat2}")

    assert (np.isclose(pose_quat1[:3], pose_quat2[:3]).all())
    assert (np.isclose(rotm1, rotm2).all())
    print("test_convert_pose passed")


def test_pose_rvec2matr_batch():
    print("===== start test_pose_rvec2matr_batch")
    poses = tf.random.uniform(shape=(8, 4, 6), minval=-1, maxval=1)
    print("input pose vector shape:", poses.get_shape())
    matrices = pose_rvec2matr_batch(poses)
    print("output pose matrix shape:", matrices.get_shape())
    pose0 = poses[3, 2, :].numpy()
    matr0 = matrices[3, 2, :, :].numpy()
    print(f"compare pose and matrix {pose0}\n{matr0}")
    assert (np.isclose(pose0[:3], matr0[:3, 3]).all())
    angle_mat = np.arccos((np.trace(matr0) - 2) / 2)
    angle_vec = np.linalg.norm(pose0[3:])
    assert (np.isclose(angle_vec, angle_mat))
    print("test_pose_rvec2matr_batch passed")


def test():
    np.set_printoptions(precision=3, suppress=True)
    test_pose_quat2mat()
    test_convert_pose()
    test_pose_rvec2matr_batch()


if __name__ == "__main__":
    test()

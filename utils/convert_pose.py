import numpy as np
import quaternion
import tensorflow as tf
import warnings
from utils.decorators import shape_check


def pose_quat2matr(pose):
    assert pose.shape[0] == 7
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


def pose_matr2quat(pose):
    trans = pose[:3, 3]
    rot_mat = pose[:3, :3].T
    quat = quaternion.from_rotation_matrix(rot_mat)
    quat = quaternion.as_float_array(quat)
    quat = quat / np.linalg.norm(quat)
    pose_quat = np.concatenate([trans, quat], axis=0)
    return pose_quat


def pose_rvec2matr_batch_tf(poses):
    """
    :param poses: poses with twist coordinates in tf.tensor, (tx, ty, tz, u1, u2, u3) [batch, N, 6]
    :return: poses in transformation matrix [batch, N, 4, 4]
    """
    # shape to [batch, N, 6, 1]
    poses = tf.expand_dims(poses, -1)
    batch, snippet, _, _ = poses.get_shape()
    # split into translation and rotation [batch, N, 3]
    trans = poses[:, :, :3]
    uvec = poses[:, :, 3:]
    # unorm: [batch, N, 1]
    unorm = tf.norm(uvec, axis=2, keepdims=True)
    uvec = uvec / unorm
    # w1.shape = [batch, N, 1, 1]
    w1 = uvec[:, :, 0:1]
    w2 = uvec[:, :, 1:2]
    w3 = uvec[:, :, 2:3]
    z = tf.zeros(shape=(batch, snippet, 1, 1))

    # w_hat.shape = [batch, N, 9, 1]
    # NOTE: 원래 책에는 이렇게 하라고 되어 있지만 이렇게 하면 반대 회전이 나옴
    # w_hat = tf.concat([z, -w3, w2, w3, z, -w1, -w2, w1, z], axis=2)
    # 회전 방향을 맞추기 위해 부호 반대로
    w_hat = tf.concat([z, w3, -w2, -w3, z, w1, w2, -w1, z], axis=2)
    # w_hat.shape = [batch, N, 3, 3]
    w_hat = tf.reshape(w_hat, shape=(batch, snippet, 3, 3))

    # identity.shape = [1, 1, 3, 3]
    identity = tf.expand_dims(tf.expand_dims(tf.eye(3), axis=0), axis=0)
    # identity.shape = [batch, N, 3, 3]
    identity = tf.tile(identity, (batch, snippet, 1, 1))
    tmpmat = identity + w_hat*tf.sin(unorm) + tf.matmul(w_hat, w_hat)*(1 - tf.cos(unorm))
    rotmat = tf.where(tf.abs(unorm) < 1e-8, identity, tmpmat)

    tmat = tf.concat([rotmat, trans], axis=3)
    last_row = tf.tile(tf.constant([[[[0, 0, 0, 1]]]], dtype=tf.float32), multiples=(batch, snippet, 1, 1))
    tmat = tf.concat([tmat, last_row], axis=2)
    tmat = tf.reshape(tmat, (batch, snippet, 4, 4))
    return tmat


def pose_rvec2matr_batch_np(poses):
    """
    :param poses: poses with twist coordinates in np.array, (tx, ty, tz, u1, u2, u3) [batch, N, 6]
    :return: poses in transformation matrix [batch, N, 4, 4]
    """
    poses = np.copy(poses)
    # shape to [batch, N, 6]
    batch, snippet, _ = poses.shape
    trans = poses[:, :, :3]
    uvec = poses[:, :, 3:]
    # unorm: [batch, N]
    unorm = np.linalg.norm(uvec, axis=2)
    unorm_mask = ~np.isclose(unorm, 0)
    uvec[unorm_mask] = uvec[unorm_mask] / unorm[unorm_mask][..., np.newaxis]
    # w1.shape = [batch, N, 1]
    w1 = uvec[:, :, 0:1]
    w2 = uvec[:, :, 1:2]
    w3 = uvec[:, :, 2:3]
    z = np.zeros(shape=(batch, snippet, 1))

    # w_hat.shape = [batch, N, 9]
    # NOTE: 원래 책에는 이렇게 하라고 되어 있지만 이렇게 하면 반대 회전이 나옴
    # w_hat = np.concatenate([z, -w3, w2, w3, z, -w1, -w2, w1, z], axis=2)
    # 회전 방향을 맞추기 위해 부호 반대로
    w_hat = np.concatenate([z, w3, -w2, -w3, z, w1, w2, -w1, z], axis=2).reshape(batch, snippet, 3, 3)
    # w_hat.shape = [batch, N, 3, 3]

    identity = np.eye(3).reshape(1, 1, 3, 3)
    identity = np.tile(identity, (batch, snippet, 1, 1))
    # identity: [batch, N, 3, 3]
    unorm = unorm.reshape(batch, snippet, 1, 1)
    rotmat = identity + w_hat*np.sin(unorm) + np.matmul(w_hat, w_hat)*(1 - np.cos(unorm))

    trans = trans.reshape(batch, snippet, 3, 1)
    tmat = np.concatenate([rotmat, trans], axis=3)
    last_row = np.tile(np.array([[[[0, 0, 0, 1]]]], dtype=np.float32), (batch, snippet, 1, 1))
    tmat = np.concatenate([tmat, last_row], axis=2)
    return tmat


def pose_rvec2matr(poses):
    """
    :param poses: poses with twist coordinates in np.array, (tx, ty, tz, u1, u2, u3) [N, 6]
    :return: poses in transformation matrix [N, 4, 4]
    """
    poses = np.copy(poses)
    poses = np.expand_dims(poses, axis=-1)
    trjlen, _, _ = poses.shape
    trans = poses[:, :3]
    uvec = poses[:, 3:]
    unorm = np.expand_dims(np.linalg.norm(uvec, axis=1), axis=1)
    # uvec [5, 3, 1], unorm [5, 1, 1]
    unorm_mask = ~np.isclose(unorm, 0).reshape(trjlen)
    uvec[unorm_mask] = uvec[unorm_mask] / unorm[unorm_mask]
    w1 = uvec[:, 0:1]
    w2 = uvec[:, 1:2]
    w3 = uvec[:, 2:3]
    z = np.zeros(shape=(trjlen, 1, 1))

    # w_hat.shape = [batch, N, 9, 1]
    # NOTE: 원래 책에는 이렇게 하라고 되어 있지만 이렇게 하면 반대 회전이 나옴
    # w_hat = np.concatenate([z, -w3, w2, w3, z, -w1, -w2, w1, z], axis=1)
    # 회전 방향을 맞추기 위해 부호 반대로
    w_hat = np.concatenate([z, w3, -w2, -w3, z, w1, w2, -w1, z], axis=1)
    w_hat = np.reshape(w_hat, (trjlen, 3, 3))
    identity = np.expand_dims(np.eye(3), axis=0)
    identity = np.tile(identity, (trjlen, 1, 1))
    # unorm = np.expand_dims(unorm, axis=-1)
    # identity: [trjlen, 3, 3], unorm: [trjlen, 1, 1], w_hat: [trjlen, 3, 3]
    rotmat = identity + w_hat*np.sin(unorm) + np.matmul(w_hat, w_hat)*(1 - np.cos(unorm))

    tmat = np.concatenate([rotmat, trans], axis=2)
    last_row = np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (trjlen, 1, 1))
    tmat = np.concatenate([tmat, last_row], axis=1)
    return tmat


def pose_matr2rvec_batch(poses):
    """ shape checked!
    :param poses: poses in transformation matrix as tf.tensor, [batch, numsrc, 4, 4]
    :return: poses with twist coordinates as tf.tensor, [batch, numsrc, 6]
    """
    # matrix에서 twist 형식으로 변환
    R = poses[:, :, :3, :3]
    theta = tf.math.acos((tf.linalg.trace(R) - 1.) / 2.)
    # theta: [batch, numsrc] -> [batch, numsrc, 1]
    theta = tf.expand_dims(theta, -1)
    # axis: [batch, numsrc, 3]
    axis = tf.stack([R[:, :, 1, 2] - R[:, :, 2, 1],
                     R[:, :, 2, 0] - R[:, :, 0, 2],
                     R[:, :, 0, 1] - R[:, :, 1, 0]], axis=-1)
    rvec = tf.where(tf.abs(theta) < 0.00001, axis / 2., axis / (2 * tf.math.sin(theta)) * theta)
    trans = poses[:, :, :3, 3]
    pose_vec = tf.concat([trans, rvec], axis=-1)
    return pose_vec


def pose_matr2rvec(poses):
    """ shape checked!
    :param poses: poses in transformation matrix as np.array, [N, 4, 4]
    :return: poses with twist coordinates as np.array, [N, 6]
    """
    # matrix에서 twist 형식으로 변환
    R = poses[:, :3, :3]
    theta = np.arccos((np.trace(R, axis1=1, axis2=2) - 1.) / 2.)
    # theta: [batch] -> [batch, 1]
    theta = theta[:, np.newaxis]
    # axis: [batch, 3]
    axis = np.stack([R[:, 1, 2] - R[:, 2, 1],
                     R[:, 2, 0] - R[:, 0, 2],
                     R[:, 0, 1] - R[:, 1, 0]], axis=-1)
    # rvec: [batch, 3]
    rvec = np.where(np.abs(theta) < 0.00001, axis / 2., axis / (2 * np.sin(theta)) * theta)
    # trans: [batch, 3]
    trans = poses[:, :3, 3]
    # pose_vec: [batch, 6]
    pose_vec = np.concatenate([trans, rvec], axis=-1)
    return pose_vec


# --------------------------------------------------------------------------------
# TESTS

def test_pose_quat2matr():
    print("===== start test_pose_quat2matr")
    rmat = pose_quat2matr(np.array([0, 0, 0, 1, 0, 0, 0]))
    assert (np.isclose(np.identity(4), rmat).all()), \
        f"[test_pose_quat2matr], np.identity(4) != \n{rmat}"
    print("!!! test_pose_quat2matr passed")


def test_convert_pose():
    print("===== start test_convert_pose")
    pose_quat1 = np.array([1, 2, 3, np.cos(np.pi/3), 0, 0, np.sin(np.pi/3)])
    pose_mat = pose_quat2matr(pose_quat1)
    pose_quat2 = pose_matr2quat(pose_mat)

    quat1 = quaternion.from_float_array(pose_quat1[3:])
    quat2 = quaternion.from_float_array(pose_quat2[3:])
    rotm1 = quaternion.as_rotation_matrix(quat1)
    rotm2 = quaternion.as_rotation_matrix(quat2)
    print(f"convert pose quat1={quat1}, quat2={quat2}")

    assert (np.isclose(pose_quat1[:3], pose_quat2[:3]).all())
    assert (np.isclose(rotm1, rotm2).all())
    print("!!! test_convert_pose passed")


def test_pose_rvec2matr_batch():
    # check converted translation and rotation angle
    print("===== start test_pose_rvec2matr_batch")
    poses_rvec = tf.random.uniform(shape=(8, 4, 6), minval=-1, maxval=1)
    print("input pose vector shape:", poses_rvec.get_shape())

    # TEST
    poses_matr = pose_rvec2matr_batch_tf(poses_rvec)

    print("output pose matrix shape:", poses_matr.get_shape())
    pose0 = poses_rvec[3, 2, :].numpy()
    matr0 = poses_matr[3, 2, :, :].numpy()
    print(f"compare poses in vector and matrix:\n{pose0} (vector)\n{matr0} (matrix)")
    # 위치 비교
    assert (np.isclose(pose0[:3], matr0[:3, 3]).all())
    # 회전 각도 비교
    angle_mat = np.arccos((np.trace(matr0[:3, :3]) - 1) / 2)
    angle_vec = np.linalg.norm(pose0[3:])
    print("angles:", angle_mat, angle_vec)
    assert (np.isclose(angle_vec, angle_mat))
    print("!!! test_pose_rvec2matr_batch passed")


def test_pose_rvec2matr():
    print("===== start test_pose_rvec2matr")
    poses = np.array([[1, 2, 3, 0, 0, np.pi/6], [4, 5, 6, np.pi/2, 0, 0]])
    # TEST
    poses_mat = pose_rvec2matr(poses)

    print(poses_mat)
    assert np.isclose(poses[:, :3], poses_mat[:, :3, 3]).all()
    print("!!! test_pose_rvec2matr passed")


def test_pose_matr2rvec_batch():
    # twist -> matrix -> twist_again -> matrix_again, check twist == twist_again ?
    print("===== start test_pose_matr2rvec_batch")
    poses_twis = tf.random.uniform(shape=(8, 4, 6), minval=-1, maxval=1)
    print("input pose vector shape:", poses_twis.get_shape())
    poses_matr = pose_rvec2matr_batch_tf(poses_twis)

    # TEST
    poses_twis_again = pose_matr2rvec_batch(poses_matr)

    print("output pose vector shape:", poses_twis_again.get_shape())
    print("original pose vector:\n", poses_twis[1, 2].numpy())
    print("converted pose matrix:\n", poses_matr[1, 2].numpy())
    print("reconstructed pose vector:\n", poses_twis_again[1, 2].numpy())
    assert np.isclose(poses_twis.numpy(), poses_twis_again.numpy()).all()
    print("!!! test_pose_matr2rvec_batch passed")


def test():
    np.set_printoptions(precision=4, suppress=True)
    test_pose_quat2matr()
    test_convert_pose()
    test_pose_rvec2matr_batch()
    test_pose_rvec2matr()
    test_pose_matr2rvec_batch()


if __name__ == "__main__":
    test()

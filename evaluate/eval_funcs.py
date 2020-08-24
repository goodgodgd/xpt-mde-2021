import numpy as np
import tensorflow as tf
import settings
import utils.convert_pose as cp
from utils.decorators import shape_check


def compute_depth_metrics(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]


def recover_pred_snippet_poses(poses):
    """
    :param poses: source poses that transforms points in target to source frame
                    format=(tx, ty, tz, ux, uy, uz) shape=[numsrc, 6]
    :return: snippet pose matrices that transforms points in source[i] frame to source[0] frame
                    format=(4x4 transformation) shape=[snippet_len, 4, 4]
                    order=[source[0], source[1], target, source[2], source[3]]
    """
    # insert origin pose as target pose into the middle
    target_pose = np.zeros(shape=(1, 6), dtype=np.float32)
    poses_vec = np.concatenate([poses[:2], target_pose, poses[2:]], axis=0)
    poses_mat = cp.pose_rvec2matr(poses_vec)
    recovered_pose = relative_pose_from_first(poses_mat)
    return recovered_pose


def recover_true_snippet_poses(poses):
    """
    :param poses: source poses that transforms points in target to source frame
                    format=(4x4 transformation), shape=[numsrc, 4, 4]
    :return: snippet pose matrices that transforms points in source[i] frame to source[0] frame
                    format=(4x4 transformation) shape=[snippet_len, 4, 4]
                    order=[source[0], source[1], target, source[2], source[3]]
    """
    target_pose = np.expand_dims(np.identity(4, dtype=np.float32), axis=0)
    poses_mat = np.concatenate([poses[:2], target_pose, poses[2:]], axis=0)
    recovered_pose = relative_pose_from_first(poses_mat)
    return recovered_pose


def relative_pose_from_first(poses_mat):
    """
    :param poses_mat: 4x4 transformation matrices, [N, 4, 4]
    :return: 4x4 transformation matrices with origin of poses_mat[0], [N, 4, 4]
    """
    poses_mat_transformed = []
    pose_origin = poses_mat[0]
    for pose_mat in poses_mat:
        # inv(source[0] to target) * (source[i] to target)
        # = (target to source[0]) * (source[i] to target)
        # = (source[i] to source[0])
        pose_mat_tfm = np.matmul(np.linalg.inv(pose_origin), pose_mat)
        poses_mat_transformed.append(pose_mat_tfm)

    poses_mat_transformed = np.stack(poses_mat_transformed, axis=0)
    return poses_mat_transformed


def calc_trajectory_error(pose_pred_mat, pose_true_mat):
    """
    :param pose_pred_mat: predicted snippet pose matrices w.r.t the first frame, [numsrc, 5, 4, 4]
    :param pose_true_mat: ground truth snippet pose matrices w.r.t the first frame, [numsrc, 5, 4, 4]
    :return: trajectory error in meter [numsrc]
    """
    xyz_pred = pose_pred_mat[:, :3, 3]
    xyz_true = pose_true_mat[:, :3, 3]
    # optimize the scaling factor
    scale = np.sum(xyz_true * xyz_pred) / np.sum(xyz_pred ** 2)
    traj_error = xyz_true - xyz_pred * scale
    rmse = np.sqrt(np.sum(traj_error ** 2, axis=1)) / len(traj_error)
    return rmse


def calc_rotational_error(pose_pred_mat, pose_true_mat):
    """
    :param pose_pred_mat: predicted snippet pose matrices w.r.t the first frame, [snippet_len, 5, 4, 4]
    :param pose_true_mat: ground truth snippet pose matrices w.r.t the first frame, [snippet_len, 5, 4, 4]
    :return: rotational error in rad [snippet_len]
    """
    rot_pred = pose_pred_mat[:, :3, :3]
    rot_true = pose_true_mat[:, :3, :3]
    rot_rela = np.matmul(np.linalg.inv(rot_pred), rot_true)
    trace = np.trace(rot_rela, axis1=1, axis2=2)
    angle = np.clip((trace - 1.) / 2., -1., 1.)
    angle = np.arccos(angle)
    return angle


@shape_check
def calc_trajectory_error_tensor(pose_pred_mat, pose_true_mat, abs_scale=False):
    """
    :param pose_pred_mat: predicted snippet pose matrices, [batch, numsrc, 4, 4]
    :param pose_true_mat: ground truth snippet pose matrices, [batch, numsrc, 4, 4]
    :param abs_scale: if true, trajectory is evaluated in abolute scale
    :return: trajectory error in meter [batch, numsrc]
    """
    xyz_pred = pose_pred_mat[:, :, :3, 3]
    xyz_true = pose_true_mat[:, :, :3, 3]
    # adjust the trajectory scaling due to ignorance of abolute scale
    scale = tf.reduce_sum(xyz_true * xyz_pred, axis=2) / tf.reduce_sum(xyz_pred ** 2, axis=2)
    traj_error = tf.cond(abs_scale, lambda: xyz_true - xyz_pred,
                         lambda: xyz_true - xyz_pred * tf.expand_dims(scale, -1))
    traj_error = tf.sqrt(tf.reduce_sum(traj_error ** 2, axis=2))
    return traj_error


@shape_check
def calc_rotational_error_tensor(pose_pred_mat, pose_true_mat):
    """
    :param pose_pred_mat: predicted snippet pose matrices w.r.t the first frame, [batch, numsrc, 4, 4]
    :param pose_true_mat: ground truth snippet pose matrices w.r.t the first frame, [batch, numsrc, 4, 4]
    :return: rotational error in rad [batch, numsrc]
    """
    rot_pred = pose_pred_mat[:, :, :3, :3]
    rot_true = pose_true_mat[:, :, :3, :3]
    rot_rela = tf.matmul(tf.linalg.inv(rot_pred), rot_true)
    trace = tf.linalg.trace(rot_rela)
    angle = tf.clip_by_value((trace - 1.) / 2., -1., 1.)
    angle = tf.math.acos(angle)
    return angle


# ==================== tests ====================

def test_calc_trajectory_error_tensor():
    # shape = [batch, numsrc, 6]
    pose_vec1 = np.random.rand(8, 4, 6) * 2. - 1.
    # pose_vec2 보다 pose_vec3에 두 배의 오차 추가
    # calc_trajectory_error_tensor() 자체에 scale 조절 기능이 있기 때문에 에러의 절대 값을 확인할 순 없음
    pose_vec2 = pose_vec1 + np.array([0, 1, 0, 0, 0, 0])
    pose_vec3 = pose_vec1 + np.array([0, 2, 0, 0, 0, 0])
    # pose_vec4는 단순히 스케일만 다르게 함
    pose_vec4 = pose_vec1
    pose_vec4[:, :, :3] *= 2.
    # tensor로 변환
    pose_vec1 = tf.constant(pose_vec1, dtype=tf.float32)
    pose_vec2 = tf.constant(pose_vec2, dtype=tf.float32)
    pose_vec3 = tf.constant(pose_vec3, dtype=tf.float32)
    pose_vec4 = tf.constant(pose_vec4, dtype=tf.float32)
    # matrix로 변환
    pose_mat1 = cp.pose_rvec2matr_batch(pose_vec1)
    pose_mat2 = cp.pose_rvec2matr_batch(pose_vec2)
    pose_mat3 = cp.pose_rvec2matr_batch(pose_vec3)
    pose_mat4 = cp.pose_rvec2matr_batch(pose_vec4)

    # TEST
    trjerr12 = calc_trajectory_error_tensor(pose_mat1, pose_mat2).numpy()
    trjerr13 = calc_trajectory_error_tensor(pose_mat1, pose_mat3).numpy()
    trjerr14 = calc_trajectory_error_tensor(pose_mat1, pose_mat4).numpy()

    # 오차가 2배인지 확인
    assert np.isclose(trjerr12 * 2, trjerr13).all()
    # 스케일만 다를 경우 오차는 0이어야 함
    assert np.isclose(trjerr14, 0).all()
    print("!!! test [calc_trajectory_error_tensor] passed")


def test_calc_rotational_error_tensor():
    # shape = [batch, numsrc, 6]
    pose_vec1 = np.random.rand(8, 4, 6) * 2. - 1.
    norms1 = np.linalg.norm(pose_vec1[:, :, 3:], axis=2, keepdims=True)
    # pose_vec2 에는 0.5, pose_vec3 에는 1 radian의 각도 오차 추가
    pose_vec2 = pose_vec1 / norms1 * (norms1 + 0.5)
    pose_vec3 = pose_vec1 / norms1 * (norms1 + 1)
    pose_vec1 = tf.constant(pose_vec1, dtype=tf.float32)
    pose_vec2 = tf.constant(pose_vec2, dtype=tf.float32)
    pose_vec3 = tf.constant(pose_vec3, dtype=tf.float32)
    # matrix로 변환
    pose_mat1 = cp.pose_rvec2matr_batch(pose_vec1)
    pose_mat2 = cp.pose_rvec2matr_batch(pose_vec2)
    pose_mat3 = cp.pose_rvec2matr_batch(pose_vec3)

    # TEST
    roterr12 = calc_rotational_error_tensor(pose_mat1, pose_mat2)
    roterr13 = calc_rotational_error_tensor(pose_mat1, pose_mat3)

    # 오차의 절대값 확인
    assert np.isclose(roterr12.numpy(), 0.5).all()
    assert np.isclose(roterr13.numpy(), 1.).all()
    print("!!! test [calc_rotational_error_tensor] passed")


def test():
    np.set_printoptions(precision=5, suppress=True)
    test_calc_trajectory_error_tensor()
    test_calc_rotational_error_tensor()


if __name__ == "__main__":
    test()

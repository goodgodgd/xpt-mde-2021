import numpy as np
import tensorflow as tf
import settings
import utils.convert_pose as cp
from utils.decorators import shape_check
from config import opts


class PoseMetricNumpy:
    def __init__(self):
        self.trj_abs_err = np.array([])
        self.trj_rel_err = np.array([])
        self.rot_err = np.array([])

    def compute_pose_errors(self, pose_pred, pose_true_mat):
        """
        :param pose_pred: 6-DoF poses [batch, numsrc, 6]
        :param pose_true_mat: 4x4 transformation matrix [batch, numsrc, 4, 4]
        self.trj_abs_err, self.trj_rel_err, self.rot_err: [batch, numsrc]
        """
        pose_pred_mat = cp.pose_rvec2matr_batch_np(pose_pred)
        pose_pred_mat = self.snippet_pose_from_first(pose_pred_mat)
        pose_true_mat = self.snippet_pose_from_first(pose_true_mat)
        self.trj_abs_err = self.calc_trajectory_error(pose_pred_mat, pose_true_mat, True)
        self.trj_rel_err = self.calc_trajectory_error(pose_pred_mat, pose_true_mat, False)
        self.rot_err = self.calc_rotational_error(pose_pred_mat, pose_true_mat)

    def snippet_pose_from_first(self, poses):
        """
        :param poses: 4x4 transformation matrices, [batch, numsrc, 4, 4]
        :return: 4x4 transformation matrices with origin of poses_mat[0], [batch, snippet, 4, 4]
        """
        target_pose = np.identity(4, dtype=np.float32).reshape(1, 1, 4, 4)
        target_pose = np.tile(target_pose, (poses.shape[0], 1, 1, 1))
        poses_mat = np.concatenate([poses[:, :2], target_pose, poses[:, 2:]], axis=1)
        poses_origin = poses_mat[:, 0:1, :, :]
        # poses_origin: [batch, 1, 4, 4], poses_mat: [batch, numsrc+1, 4, 4]
        poses_mat_tfm = np.matmul(np.linalg.inv(poses_origin), poses_mat)
        return poses_mat_tfm

    @shape_check
    def calc_trajectory_error(self, pose_pred_mat, pose_true_mat, abs_scale=False):
        """
        :param pose_pred_mat: predicted snippet pose matrices, [batch, snippet, 4, 4]
        :param pose_true_mat: ground truth snippet pose matrices, [batch, snippet, 4, 4]
        :param abs_scale: if true, trajectory is evaluated in abolute scale
        :return: trajectory error in meter [batch, snippet-1]
        """
        xyz_pred = pose_pred_mat[:, :, :3, 3]
        xyz_true = pose_true_mat[:, :, :3, 3]
        # adjust the trajectory scaling due to ignorance of abolute scale
        if abs_scale:
            traj_error = xyz_true - xyz_pred
        else:
            scale = np.sum(xyz_true * xyz_pred, axis=2) / np.sum(xyz_pred ** 2, axis=2)
            traj_error = xyz_true - xyz_pred * scale[..., np.newaxis]
        traj_error = np.sqrt(np.sum(traj_error ** 2, axis=2))
        traj_error = traj_error[:, 1:]
        return traj_error

    @shape_check
    def calc_rotational_error(self, pose_pred_mat, pose_true_mat):
        """
        :param pose_pred_mat: predicted snippet pose matrices w.r.t the first frame, [batch, snippet, 4, 4]
        :param pose_true_mat: ground truth snippet pose matrices w.r.t the first frame, [batch, snippet, 4, 4]
        :return: rotational error in rad [batch, snippet-1]
        """
        rot_pred = pose_pred_mat[:, :, :3, :3]
        rot_true = pose_true_mat[:, :, :3, :3]
        rot_rela = np.matmul(np.linalg.inv(rot_pred), rot_true)
        trace = np.trace(rot_rela, axis1=2, axis2=3)
        angle = np.clip((trace - 1.) / 2., -1., 1.)
        angle = np.arccos(angle)
        angle = angle[:, 1:]
        return angle

    def get_mean_pose_error(self):
        return np.mean(self.trj_abs_err), np.mean(self.trj_rel_err), np.mean(self.rot_err)

    def save_all_txt(self):
        pass

    def save_summary_csv(self):
        pass


class PoseMetricTf(PoseMetricNumpy):
    """
    Pose metric evaluator with tf.tensor data
    """
    def __init__(self):
        super().__init__()

    def compute_pose_errors(self, pose_pred, pose_true_mat):
        """
        :param pose_pred: 6-DoF poses [batch, numsrc, 6]
        :param pose_true_mat: 4x4 transformation matrix [batch, numsrc, 4, 4]
        """
        pose_pred = pose_pred.numpy()
        pose_true_mat = pose_true_mat.numpy()
        pose_pred_mat = cp.pose_rvec2matr_batch_np(pose_pred)
        pose_pred_mat = self.snippet_pose_from_first(pose_pred_mat)
        pose_true_mat = self.snippet_pose_from_first(pose_true_mat)
        self.trj_abs_err = self.calc_trajectory_error(pose_pred_mat, pose_true_mat, True)
        self.trj_rel_err = self.calc_trajectory_error(pose_pred_mat, pose_true_mat, False)
        self.rot_err = self.calc_rotational_error(pose_pred_mat, pose_true_mat)


def valid_depth_filter(depth_pred, depth_true):
    """
    :param depth_pred: [height, width]
    :param depth_true: [height, width]
    :return: depth_pred, depth_true: [N]
    """
    depth_pred = np.squeeze(depth_pred)
    depth_true = np.squeeze(depth_true)
    mask = np.logical_and(depth_true > opts.MIN_DEPTH, depth_true < opts.MAX_DEPTH)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = depth_true.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    # scale matching, depth_true[mask].shape = (batch, N)
    scaler = np.median(depth_true[mask]) / np.median(depth_pred[mask])
    depth_pred[mask] *= scaler
    # clip prediction and compute error metrics
    depth_pred = np.clip(depth_pred, opts.MIN_DEPTH, opts.MAX_DEPTH)
    return depth_pred[mask], depth_true[mask]


def compute_depth_metrics(pred, gt):
    """
    :param pred: [N]
    :param gt: [N]
    """
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


# ==================== tests ====================

def test_PoseMetricNumpy_trjerr():
    # shape = [batch, numsrc, 6]
    pose_vec1 = np.random.rand(8, 4, 6) * 2. - 1.
    # pose_vec2 보다 pose_vec3에 두 배의 오차 추가
    # calc_trajectory_error_tensor() 자체에 scale 조절 기능이 있기 때문에 에러의 절대 값을 확인할 순 없음
    pose_vec2 = pose_vec1.copy()
    pose_vec2[:, 1:, :] = pose_vec1[:, 1:, :] + np.array([0, 1, 0, 0, 0, 0])
    pose_vec3 = pose_vec1.copy()
    pose_vec3[:, 1:, :] = pose_vec1[:, 1:, :] + np.array([0, 2, 0, 0, 0, 0])
    # pose_vec4는 단순히 스케일만 다르게 함
    pose_vec4 = pose_vec1.copy()
    pose_vec4[:, :, :3] *= 2.
    # tensor로 변환
    pose_vec1 = tf.constant(pose_vec1, dtype=tf.float32)
    pose_vec2 = tf.constant(pose_vec2, dtype=tf.float32)
    pose_vec3 = tf.constant(pose_vec3, dtype=tf.float32)
    pose_vec4 = tf.constant(pose_vec4, dtype=tf.float32)
    # matrix로 변환
    pose_mat1 = cp.pose_rvec2matr_batch_tf(pose_vec1)
    pose_mat2 = cp.pose_rvec2matr_batch_tf(pose_vec2)
    pose_mat3 = cp.pose_rvec2matr_batch_tf(pose_vec3)
    pose_mat4 = cp.pose_rvec2matr_batch_tf(pose_vec4)

    # TEST
    eval12 = PoseMetricTf()
    eval12.compute_pose_errors(pose_vec1, pose_mat2)
    eval13 = PoseMetricTf()
    eval13.compute_pose_errors(pose_vec1, pose_mat3)
    eval14 = PoseMetricTf()
    eval14.compute_pose_errors(pose_vec1, pose_mat4)

    # 오차가 2배인지 확인
    assert np.isclose(eval12.trj_abs_err * 2., eval13.trj_abs_err, atol=1e-5).all()
    # 스케일만 다를 경우 오차는 0이어야 함
    assert np.isclose(eval14.trj_rel_err, 0, atol=1e-5).all()
    print("!!! test [test_PoseMetricNumpy_roterr] passed")


def test_PoseMetricNumpy_roterr():
    # shape = [batch, numsrc, 6]
    # snippet 내부 모든 회전이 크기는 1인 동일한 회전이 되도록 설정
    pose_vec1 = np.random.rand(8, 4, 6) * 2. - 1.
    pose_vec1[:, 1:, 3:] = pose_vec1[:, 0:1, 3:]
    norms1 = np.linalg.norm(pose_vec1[:, 0:1, 3:], axis=2, keepdims=True)
    pose_vec1[:, :, 3:] = pose_vec1[:, :, 3:] / norms1
    # snippet 회전크기가 0.5, 1, 1.5 가 되도록 설정
    pose_vec2 = pose_vec1.copy()
    pose_vec2[:, 1, 3:] *= 0.5
    pose_vec2[:, 3, 3:] *= 1.5
    pose_vec1 = tf.constant(pose_vec1, dtype=tf.float32)
    pose_vec2 = tf.constant(pose_vec2, dtype=tf.float32)
    # matrix로 변환
    pose_mat1 = cp.pose_rvec2matr_batch_tf(pose_vec1)
    pose_mat2 = cp.pose_rvec2matr_batch_tf(pose_vec2)
    eval12 = PoseMetricTf()
    eval12.compute_pose_errors(pose_vec1, pose_mat2)
    print("eval12.rot_err\n", eval12.rot_err)
    assert np.isclose(eval12.rot_err[:, 0], 0.5).all()
    assert np.isclose(eval12.rot_err[:, 1], 0., atol=0.001).all()
    assert np.isclose(eval12.rot_err[:, 2], 0., atol=0.001).all()
    assert np.isclose(eval12.rot_err[:, 3], 0.5).all()
    print("!!! test [test_PoseMetricNumpy_roterr] passed")
    return


def test():
    np.set_printoptions(precision=5, suppress=True)
    test_PoseMetricNumpy_trjerr()
    test_PoseMetricNumpy_roterr()


if __name__ == "__main__":
    test()

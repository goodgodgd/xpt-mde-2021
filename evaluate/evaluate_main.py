import os
import os.path as op
import numpy as np

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
from utils.util_funcs import pose_rvec2matr


def evaluate(test_dir_name, model_dir):
    total_depth_pred, total_pose_pred = load_predictions(model_dir)
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name), batch_size=1).get_generator()
    depth_errors = []
    trajectory_errors = []
    rotational_errors = []

    for i, (x, y) in enumerate(dataset):
        if i >= total_pose_pred.shape[0]:
            break
        print("===== index:", i)
        depth_true = x["depth_gt"].numpy()[0]
        pose_true = x["pose_gt"].numpy()[0]
        depth_pred = total_depth_pred[i]
        pose_pred = total_pose_pred[i]

        depth_err = evaluate_depth(depth_pred, depth_true)
        trj_err, rot_err = evaluate_pose(pose_pred, pose_true)
        depth_errors.append(depth_err)
        trajectory_errors.append(trj_err)
        rotational_errors.append(rot_err)

    depth_errors = np.array(depth_errors)
    trajectory_errors = np.array(trajectory_errors)
    rotational_errors = np.array(rotational_errors)
    print("depth error shape:", depth_errors.shape)
    print(depth_errors[:5])
    print("trajectory error shape:", trajectory_errors.shape)
    print(trajectory_errors[:5])
    print("rotational error shape:", rotational_errors.shape)
    print(rotational_errors[:5])


def load_predictions(model_dir):
    pred_dir_path = op.join(opts.DATAPATH_PRD, model_dir)
    os.makedirs(pred_dir_path, exist_ok=True)
    depth_pred = np.load(op.join(pred_dir_path, "depth.npy"))
    print(f"[load_predictions] load depth from {pred_dir_path}, shape={depth_pred.shape}")
    pose_pred = np.load(op.join(pred_dir_path, "pose.npy"))
    print(f"[load_predictions] load pose from {pred_dir_path}, shape={pose_pred.shape}")
    return depth_pred, pose_pred


def evaluate_depth(depth_pred, depth_true):
    mask = np.logical_and(depth_true > opts.MIN_DEPTH, depth_true < opts.MAX_DEPTH)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width, _ = depth_true.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    # scale matching
    scalor = np.median(depth_true[mask]) / np.median(depth_pred[mask])
    depth_pred[mask] *= scalor
    # clip prediction and compute error metrics
    depth_pred = np.clip(depth_pred, opts.MIN_DEPTH, opts.MAX_DEPTH)
    metrics = compute_errors(depth_true[mask], depth_pred[mask])
    return metrics


def compute_errors(gt, pred):
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


def evaluate_pose(pose_pred, pose_true):
    """
    :param pose_pred: predicted source poses that transforms points in target to source frame
                    format=(tx, ty, tz, ux, uy, uz) shape=[num_src, 6]
    :param pose_true: ground truth source poses that transforms points in target to source frame
                    format=(4x4 transformation), shape=[num_src, 4, 4]
    """
    # convert source and target poses to relative poses w.r.t first source pose
    # in 4x4 transformation matrix form
    pose_pred_mat = recover_pred_snippet_poses(pose_pred)
    pose_true_mat = recover_true_snippet_poses(pose_true)

    trj_error = calc_trajectory_error(pose_pred_mat, pose_true_mat)
    rot_error = calc_rotational_error(pose_pred_mat, pose_true_mat)
    return trj_error, rot_error


def recover_pred_snippet_poses(poses):
    """
    :param poses: source poses that transforms points in target to source frame
                    format=(tx, ty, tz, ux, uy, uz) shape=[num_src, 6]
    :return: snippet pose matrices that transforms points in source[i] frame to source[0] frame
                    format=(4x4 transformation) shape=[snippet_len, 5, 4, 4]
                    order=[source[0], source[1], target, source[2], source[3]]
    """
    target_pose = np.zeros(shape=(1, 6), dtype=np.float32)
    poses_vec = np.concatenate([poses[:2], target_pose, poses[2:]], axis=0)
    poses_mat = pose_rvec2matr(poses_vec)
    recovered_pose = relative_pose_from_first(poses_mat)
    return recovered_pose


def recover_true_snippet_poses(poses):
    """
    :param poses: source poses that transforms points in target to source frame
                    format=(4x4 transformation), shape=[snippet_len, 4, 4, 4]
    :return: snippet pose matrices that transforms points in source[i] frame to source[0] frame
                    format=(4x4 transformation) shape=[snippet_len, 5, 4, 4]
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
    :param pose_pred_mat: predicted snippet pose matrices w.r.t the first frame, [snippet_len, 5, 4, 4]
    :param pose_true_mat: ground truth snippet pose matrices w.r.t the first frame, [snippet_len, 5, 4, 4]
    :return: trajectory error in meter [snippet_len]
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
    angle = np.arccos((trace - 1.) / 2.)
    return angle


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    evaluate("kitti_raw_test", "vode_model")

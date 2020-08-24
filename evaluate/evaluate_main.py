import os
import os.path as op
import numpy as np

import settings
from config import opts
import evaluate.eval_funcs as ef


def evaluate_by_plan():
    for dataset_name, save_keys in opts.TEST_PLAN:
        evaluate(dataset_name)


def evaluate(dataset_name, pred_path=opts.DATAPATH_PRD, ckpt_name=opts.CKPT_NAME):
    filename = op.join(pred_path, ckpt_name, dataset_name + ".npz")
    results = np.load(filename)
    results = {key: results[key] for key in results.files}
    os.makedirs(op.join(opts.DATAPATH_EVL, ckpt_name), exist_ok=True)

    if "pose" in results and "pose_gt" in results:
        trj_errors = []
        rot_errors = []
        for pose_pred, pose_true in zip(results["pose"], results["pose_gt"]):
            print("pose shape", pose_pred.shape, pose_true.shape)
            trj_err, rot_err = evaluate_pose(pose_pred, pose_true)
            trj_errors.append(trj_err)
            rot_errors.append(rot_err)

        print("")
        trj_errors = np.stack(trj_errors, axis=0)
        rot_errors = np.stack(rot_errors, axis=0)
        print(f"trajectory errors: {trj_errors.shape}\n{trj_errors[:5]}\n-> mean={np.mean(trj_errors, axis=0)}")
        print(f"rotational errors: {rot_errors.shape}\n{rot_errors[:5]}\n-> mean={np.mean(rot_errors, axis=0)}")
        os.makedirs(op.join(opts.DATAPATH_EVL, ckpt_name), exist_ok=True)
        np.savetxt(op.join(opts.DATAPATH_EVL, ckpt_name, "trajectory_error.txt"), trj_errors, fmt="%1.4f")
        np.savetxt(op.join(opts.DATAPATH_EVL, ckpt_name, "rotation_error.txt"), rot_errors, fmt="%1.4f")

    if "depth" in results and "depth_gt" in results:
        depth_errors = []
        for depth_pred, depth_true in zip(results["depth"], results["depth_gt"]):
            depth_err = evaluate_depth(depth_pred, depth_true)
            depth_errors.append(depth_err)

        depth_errors = np.array(depth_errors)
        print(f"depth errors: {depth_errors.shape}\n{depth_errors[:5]}\n-> mean={np.mean(depth_errors, axis=0)}")
        np.savetxt(op.join(opts.DATAPATH_EVL, ckpt_name, "depth_error.txt"), depth_errors, fmt="%1.4f")


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
    scaler = np.median(depth_true[mask]) / np.median(depth_pred[mask])
    depth_pred[mask] *= scaler
    # clip prediction and compute error metrics
    depth_pred = np.clip(depth_pred, opts.MIN_DEPTH, opts.MAX_DEPTH)
    metrics = ef.compute_depth_metrics(depth_true[mask], depth_pred[mask])
    return metrics


def evaluate_pose(pose_pred, pose_true):
    """
    :param pose_pred: predicted source poses that transforms points in target to source frame
                    format=(tx, ty, tz, ux, uy, uz) shape=[numsrc, 6]
    :param pose_true: ground truth source poses that transforms points in target to source frame
                    format=(4x4 transformation), shape=[numsrc, 4, 4]
    """
    # convert source and target poses to relative poses w.r.t first source pose
    # in 4x4 transformation matrix form
    pose_pred_mat = ef.recover_pred_snippet_poses(pose_pred)
    pose_true_mat = ef.recover_true_snippet_poses(pose_true)

    trj_error = ef.calc_trajectory_error(pose_pred_mat, pose_true_mat)
    rot_error = ef.calc_rotational_error(pose_pred_mat, pose_true_mat)
    return trj_error, rot_error


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    evaluate_by_plan()

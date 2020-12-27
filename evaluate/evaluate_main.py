import os
import os.path as op
import numpy as np
import pandas as pd

import settings
from config import opts
import evaluate.eval_utils as eu
import utils.util_funcs as uf


def evaluate_by_plan():
    for dataset_name, save_keys in opts.TEST_PLAN:
        evaluate_dataset(dataset_name)


def evaluate_dataset(dataset_name, ckpt_name=opts.CKPT_NAME):
    filename = op.join(opts.DATAPATH_PRD, ckpt_name, dataset_name + ".npz")
    results = np.load(filename)
    results = {key: results[key] for key in results.files}
    os.makedirs(op.join(opts.DATAPATH_EVL, ckpt_name), exist_ok=True)

    if "pose" in results and "pose_gt" in results:
        trj_abs, rot_abs = evaluate_dataset_pose(results, True)
        trj_rel, rot_rel = evaluate_dataset_pose(results, False)
        save_pose_metric(ckpt_name, dataset_name, trj_abs, rot_abs, trj_rel, rot_rel)

    if "depth" in results and "depth_gt" in results:
        evaluate_dataset_depth(results, ckpt_name, dataset_name)


def evaluate_dataset_pose(results, abs_scale):
    num_samples = results["pose"].shape[0]
    trj_errors = np.zeros((num_samples, 4), dtype=float)
    rot_errors = np.zeros((num_samples, 4), dtype=float)
    for i, (pose_pred, pose_true) in enumerate(zip(results["pose"], results["pose_gt"])):
        trj_err, rot_err = evaluate_pose(pose_pred, pose_true, abs_scale)
        trj_errors[i, :] = trj_err[1:]
        rot_errors[i, :] = rot_err[1:]
        uf.print_progress_status(f"{i} / {num_samples}")

    print("")
    absrel = "absolute" if abs_scale else "relative"
    print(f"trajectory errors in {absrel} scale: {trj_errors.shape}")
    print(f"-> mean={np.mean(trj_errors, axis=0)}, total mean={np.mean(trj_errors):1.5f}")
    print(f"rotational errors: {rot_errors.shape}")
    print(f"-> mean={np.mean(rot_errors, axis=0)}, total mean={np.mean(rot_errors):1.5f}")
    return trj_errors, rot_errors


def save_pose_metric(ckpt_name, dataset_name, trj_abs, rot_abs, trj_rel, rot_rel):
    dstpath = op.join(opts.DATAPATH_EVL, ckpt_name)
    os.makedirs(dstpath, exist_ok=True)
    trj_errors = np.concatenate([trj_abs, trj_rel], axis=1)
    rot_errors = np.concatenate([rot_abs, rot_rel], axis=1)
    np.savetxt(op.join(dstpath, f"{dataset_name}_trjerr.txt"), trj_errors, fmt="%1.5f")
    np.savetxt(op.join(dstpath, f"{dataset_name}_roterr.txt"), rot_errors, fmt="%1.5f")
    results = {"trjmean_abs": [np.mean(trj_abs)], "trjstd_abs": [np.std(trj_abs)],
               "trjmean_rel": [np.mean(trj_rel)], "trjstd_rel": [np.std(trj_rel)],
               "rotmean_abs": [np.mean(rot_abs)], "rotstd_abs": [np.std(rot_abs)],
               "rotmean_rel": [np.mean(rot_rel)], "rotstd_rel": [np.std(rot_rel)],}
    results = pd.DataFrame(results)
    print("pose eval result:\n", results)
    results.to_csv(op.join(dstpath, f"{dataset_name}_pose_eval.csv"), index=False, float_format='%1.5f')


def evaluate_dataset_depth(results, ckpt_name, dataset_name):
    depth_errors = []
    for depth_pred, depth_true in zip(results["depth"], results["depth_gt"]):
        depth_err = evaluate_depth(depth_pred, depth_true)
        depth_errors.append(depth_err)

    depth_errors = np.array(depth_errors)
    print(f"depth errors: {depth_errors.shape}\n{depth_errors[:5]}\n-> mean={np.mean(depth_errors, axis=0)}")
    dstpath = op.join(opts.DATAPATH_EVL, ckpt_name)
    np.savetxt(op.join(dstpath, f"{dataset_name}_depth.txt"), depth_errors, fmt="%1.4f")
    results = pd.DataFrame(np.mean(depth_errors, axis=0)[np.newaxis, ...], columns=["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"])
    print("depth eval result:\n", results)
    results.to_csv(op.join(dstpath, f"{dataset_name}_depth_eval.csv"), index=False, float_format='%1.5f')


def evaluate_depth(depth_pred, depth_true):
    eu.valid_depth_filter(depth_pred, depth_true)


def evaluate_pose(pose_pred, pose_true, abs_scale):
    """
    :param pose_pred: predicted source poses that transforms points in target to source frame
                    format=(tx, ty, tz, ux, uy, uz) shape=[numsrc, 6]
    :param pose_true: ground truth source poses that transforms points in target to source frame
                    format=(4x4 transformation), shape=[numsrc, 4, 4]
    :param abs_scale: evaluate pose in absolute scale
    """
    # convert source and target poses to relative poses w.r.t first source pose
    # in 4x4 transformation matrix form
    pose_pred_mat = ef.recover_pred_snippet_poses(pose_pred)
    pose_true_mat = ef.recover_true_snippet_poses(pose_true)

    trj_error = ef.calc_trajectory_error(pose_pred_mat, pose_true_mat, abs_scale)
    rot_error = ef.calc_rotational_error(pose_pred_mat, pose_true_mat)
    return trj_error, rot_error


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    evaluate_by_plan()

import os
import os.path as op
import numpy as np

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordReader
import utils.util_funcs as uf
import evaluate.eval_funcs as ef


def evaluate_by_user_interaction():
    options = {"data_dir_name": "kitti_raw_test",
               "model_name": "vode_model",
               }

    print("\n===== Select evaluation options")

    print(f"Default options:")
    for key, value in options.items():
        print(f"\t{key} = {value}")
    print("\nIf you are happy with default options, please press enter")
    print("Otherwise, please press any other key")
    select = input()

    if select == "":
        print(f"You selected default options.")
    else:
        message = "Type 1 or 2 to specify dataset: 1) kitti_raw_test, 2) kitti_odom_test"
        ds_id = uf.input_integer(message, 1, 2)
        if ds_id == 1:
            options["data_dir_name"] = "kitti_raw_test"
        if ds_id == 2:
            options["data_dir_name"] = "kitti_odom_test"

        print("Type model_name: dir name under opts.DATAPATH_CKP and opts.DATAPATH_PRD")
        options["model_name"] = input()

    print("Prediction options:", options)
    evaluate(**options)


def evaluate(data_dir_name, model_name):
    total_depth_pred, total_pose_pred = load_predictions(model_name)
    dataset = TfrecordReader(op.join(opts.DATAPATH_TFR, data_dir_name), batch_size=1).get_dataset()
    depth_valid = uf.check_tfrecord_including(op.join(opts.DATAPATH_TFR, data_dir_name), ["depth_gt"])
    if not uf.check_tfrecord_including(op.join(opts.DATAPATH_TFR, data_dir_name), ["pose_gt"]):
        print("Evaluation is NOT possible without pose_gt")
        return

    depth_errors = []
    trajectory_errors = []
    rotational_errors = []

    for i, x in enumerate(dataset):
        if i >= total_pose_pred.shape[0]:
            break
        uf.print_numeric_progress(i, 0)
        pose_true = x["pose_gt"].numpy()[0]
        pose_pred = total_pose_pred[i]
        trj_err, rot_err = evaluate_pose(pose_pred, pose_true)
        trajectory_errors.append(trj_err)
        rotational_errors.append(rot_err)

        if depth_valid:
            depth_true = x["depth_gt"].numpy()[0]
            depth_pred = total_depth_pred[i]
            depth_err = evaluate_depth(depth_pred, depth_true)
            depth_errors.append(depth_err)

    print("")
    trajectory_errors = np.array(trajectory_errors)
    rotational_errors = np.array(rotational_errors)
    print(f"trajectory error shape: {trajectory_errors.shape}\n{trajectory_errors[:5]}")
    print(f"rotational error shape: {rotational_errors.shape}\n{rotational_errors[:5]}")
    os.makedirs(op.join(opts.DATAPATH_EVL, model_name), exist_ok=True)
    np.savetxt(op.join(opts.DATAPATH_EVL, model_name, "trajectory_error.txt"), trajectory_errors, fmt="%1.4f")
    np.savetxt(op.join(opts.DATAPATH_EVL, model_name, "rotation_error.txt"), rotational_errors, fmt="%1.4f")

    if depth_valid:
        depth_errors = np.array(depth_errors)
        print(f"depth error shape: {depth_errors.shape}\n{depth_errors[:5]}")
        np.savetxt(op.join(opts.DATAPATH_EVL, model_name, "depthe_error.txt"), depth_errors, fmt="%1.4f")


def load_predictions(model_name):
    pred_dir_path = op.join(opts.DATAPATH_PRD, model_name)
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
    evaluate('kitti_raw_test', 'vode1')

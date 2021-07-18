import os
import os.path as op
import numpy as np
import pandas as pd
from glob import glob

import settings
from config import opts
import evaluate.eval_utils as eu
import utils.util_class as uc


def evaluate_by_plan():
    for net_names, dataset_name, save_keys, ckpt_name, weight_suffix in opts.TEST_PLAN:
        evaluate_dataset(dataset_name, ckpt_name, weight_suffix)
    merge_eval_results()


def evaluate_dataset(dataset_name, ckpt_name, weight_suffix):
    eval_dir_path = op.join(opts.DATAPATH_EVL, ckpt_name)
    if op.isdir(eval_dir_path):
        print("\n[evaluate_dataset] evaluation was made in:", eval_dir_path)
        return

    with uc.PathManager([eval_dir_path], None) as pm:
        filename = op.join(opts.DATAPATH_PRD, ckpt_name, f"{dataset_name}_{weight_suffix}.npz")
        results = np.load(filename)
        results = {key: results[key] for key in results.files}

        if "pose" in results and "pose_gt" in results:
            evaluate_dataset_pose(results, ckpt_name, dataset_name, weight_suffix)
    
        if "depth" in results and "depth_gt" in results:
            evaluate_dataset_depth(results, ckpt_name, dataset_name, weight_suffix)

        pm.set_ok()


def evaluate_dataset_pose(results, ckpt_name, dataset_name, weight_suffix):
    eval_pose = eu.PoseMetricNumpy()
    eval_pose.compute_pose_errors(results["pose"], results["pose_gt"])
    print("snippet mean (trjabs, trjrel, rot):")
    print(np.mean(eval_pose.trj_abs_err, axis=0))
    print(np.mean(eval_pose.trj_rel_err, axis=0))
    print(np.mean(eval_pose.rot_err, axis=0))

    dstpath = op.join(opts.DATAPATH_EVL, ckpt_name)
    os.makedirs(dstpath, exist_ok=True)
    pose_errors = np.concatenate([eval_pose.trj_abs_err, eval_pose.trj_rel_err, eval_pose.rot_err], axis=1)
    np.savetxt(op.join(dstpath, f"pose_{dataset_name}_{weight_suffix}.txt"), pose_errors, fmt="%1.5f")
    results = {"trjmean_abs": [np.mean(eval_pose.trj_abs_err)], "trjstd_abs": [np.std(eval_pose.trj_abs_err)],
               "trjmean_rel": [np.mean(eval_pose.trj_rel_err)], "trjstd_rel": [np.std(eval_pose.trj_rel_err)],
               "rotmean": [np.mean(eval_pose.rot_err)], "rotstd": [np.std(eval_pose.rot_err)],
               }
    results = pd.DataFrame(results)
    print("pose eval result:\n", results)
    results.to_csv(op.join(dstpath, f"pose_eval_{dataset_name}_{weight_suffix}.csv"), index=False, float_format='%1.5f')


def evaluate_dataset_depth(results, ckpt_name, dataset_name, weight_suffix):
    depth_metrics = []
    for depth_pred, depth_true in zip(results["depth"], results["depth_gt"]):
        dep_metrics = evaluate_depth(depth_pred, depth_true)
        depth_metrics.append(dep_metrics)

    depth_metrics = np.array(depth_metrics)
    mean_metrics = np.mean(depth_metrics, axis=0)
    print(f"depth errors: {depth_metrics.shape}\n-> mean={mean_metrics}")
    dstpath = op.join(opts.DATAPATH_EVL, ckpt_name)
    np.savetxt(op.join(dstpath, f"depth_{dataset_name}_{weight_suffix}.txt"), depth_metrics, fmt="%1.5f")
    results = pd.DataFrame(mean_metrics[np.newaxis, ...], columns=["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"])
    print("depth eval result:\n", results)
    results.to_csv(op.join(dstpath, f"depth_eval_{dataset_name}_{weight_suffix}.csv"), index=False, float_format='%1.5f')


def evaluate_depth(depth_pred, depth_true):
    depth_pred, depth_true = eu.valid_depth_filter(depth_pred, depth_true)
    depth_metrics = eu.compute_depth_metrics(depth_pred, depth_true)
    return depth_metrics


def merge_eval_results():
    files = glob(f"{opts.DATAPATH_EVL}/*/*.csv")
    if not files:
        print("[merge_eval_results] there is NO eval result")
        return

    eval_results = []
    for file in files:
        result = pd.read_csv(file)
        result["name"] = op.basename(op.dirname(file))
        eval_results.append(result)
    eval_results = pd.concat(eval_results, axis=0)
    print("evaluation results:\n", eval_results)
    result_file = op.join(opts.DATAPATH_EVL, "merged_result.csv")
    eval_results.to_csv(result_file, index=False, float_format='%1.5f')


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    evaluate_by_plan()

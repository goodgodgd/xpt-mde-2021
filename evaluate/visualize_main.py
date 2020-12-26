import os
import os.path as op
import tensorflow as tf
import cv2
import open3d as o3d
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import settings
from tfrecords.tfrecord_reader import TfrecordReader
from config import opts
import utils.convert_pose as cp
import evaluate.eval_funcs as ef


def show_poses():
    for dataset_name, outkey in opts.TEST_PLAN:
        print(op.join(opts.DATAPATH_PRD, opts.CKPT_NAME, dataset_name + ".npz"))
        results = np.load(op.join(opts.DATAPATH_PRD, opts.CKPT_NAME, dataset_name + ".npz"))
        if "pose" in outkey:
            print_odometry_results(results)


def show_depths():
    for dataset_name, outkey in opts.TEST_PLAN:
        print(op.join(opts.DATAPATH_PRD, opts.CKPT_NAME, dataset_name + ".npz"))
        results = np.load(op.join(opts.DATAPATH_PRD, opts.CKPT_NAME, dataset_name + ".npz"))
        if "depth" in outkey:
            visualize_point_cloud(results)


def print_odometry_results(results):
    num_samples = 100
    poses_pred = results["pose"]
    poses_true = results["pose_gt"]
    stride = max(poses_pred.shape[0] // num_samples, 1)
    print("pose shape and stride:", poses_pred.shape, stride)
    for index in np.arange(0, poses_pred.shape[0], stride):
        pose_pr_tws = poses_pred[index]
        pose_pr_mat = cp.pose_rvec2matr(pose_pr_tws)
        pose_gt_mat = poses_true[index]
        pose_gt_tws = cp.pose_matr2rvec(pose_gt_mat)
        trjerr_rel = ef.calc_trajectory_error(pose_pr_mat, pose_gt_mat)[..., np.newaxis]
        trjerr_abs = ef.calc_trajectory_error(pose_pr_mat, pose_gt_mat, True)[..., np.newaxis]
        roterr = ef.calc_rotational_error(pose_pr_mat, pose_gt_mat)[..., np.newaxis]
        view = np.concatenate([pose_gt_tws, pose_pr_tws, trjerr_rel, trjerr_abs, roterr], axis=1)
        print(f"pose result at {index}: [pose_gt_tws, pose_pr_tws, trjerr_rel, trjerr_abs, roterr]\n{view}")


def visualize_point_cloud(results):
    num_samples = 100
    depths_pred = results["depth"]
    depths_true = results["depth_gt"]
    Ks = results["intrinsic"]
    stride = max(depths_pred.shape[0] // num_samples, 1)
    print("depth shape and stride:", depths_pred.shape, depths_true.shape, stride)
    for index in np.arange(0, depths_true.shape[0], stride):
        K = Ks[index]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        points_pr = to_point_cloud(depths_pred[index], K, [1, 0, 0])
        points_gt = to_point_cloud(depths_true[index], K, [0, 1, 0])
        o3d.visualization.draw_geometries([points_gt, points_pr, frame])
        # o3d.visualization.draw_geometries([points_gt, points_pr])


def to_point_cloud(depth, K, color):
    H, W, _ = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1)
    v = v.reshape(-1)
    Z = depth.reshape(-1)
    X = (u - cx)/fx * Z
    Y = (v - cy)/fy * Z
    points = np.stack([X, Y, Z], axis=1)
    points = points[(Z > 0) & (Z < 40) & (Y > -2.5) & (Y < 1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return pcd


def visualize(data_dir_name, model_name):
    depths = np.load(op.join(opts.DATAPATH_PRD, model_name, "depth.npy"))
    poses = np.load(op.join(opts.DATAPATH_PRD, model_name, "pose.npy"))
    print(f"depth shape: {depths.shape}, pose shape: {poses.shape}")
    tfrgen = TfrecordReader(op.join(opts.DATAPATH_TFR, data_dir_name), batch_size=1)
    dataset = tfrgen.get_dataset()
    fig = plt.figure()
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)

    for i, (x, y) in enumerate(dataset):
        image = tf.image.convert_image_dtype((x["image"] + 1.)/2., dtype=tf.uint8)
        image = image[0].numpy()

        depth = np.squeeze(depths[i], axis=-1)
        pose_snippet = poses[i]
        print("source frame poses w.r.t target (center) frame")
        print(pose_snippet)

        cv2.imshow("image", image)
        cv2.waitKey(1000)
        print("depth", depth.shape)
        plt.imshow(depth, cmap="viridis")
        plt.show()


if __name__ == "__main__":
    show_poses()
    show_depths()

import os
import os.path as op
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordReader
import utils.util_funcs as uf
import utils.convert_pose as cp
import evaluate.eval_funcs as ef
from model.synthesize.synthesize_base import SynthesizeMultiScale
from model.loss_and_metric import photometric_loss, smootheness_loss
from model.model_main import set_configs, create_model, try_load_weights


def evaluate_for_debug(data_dir_name, model_name):
    """
    function to check if learning process is going right
    to evaluate current model, save losses and error metrics to csv files and save debugging images
    - debug_depth.csv: 타겟 프레임별로 predicted depth의 error와 smootheness loss 저장
    - debug_pose.csv: 소스 프레임별로 photometric loss, trajectory error, rotation error 저장
    - trajectory.csv: 소스 프레임별로 gt trajectory, pred trajectory 저장
    - debug_imgs(directory): loss와 metric 별로 가장 성능이 안좋은 프레임들을 모아서 inspection view 이미지로 저장
        1) target image
        2) reconstructed target from gt
        3) reconstructed target from pred
        4) source image
        5) predicted target depth
    """
    if not uf.check_tfrecord_including(op.join(opts.DATAPATH_TFR, data_dir_name), ["pose_gt", "depth_gt"]):
        print("Evaluation is NOT possible without pose_gt and depth_gt")
        return

    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset = TfrecordReader(op.join(opts.DATAPATH_TFR, data_dir_name), batch_size=1).get_dataset()
    depth_result = []
    pose_result = []
    trajectory = []
    steps_per_epoch = uf.count_steps(data_dir_name, 1)

    for i, x in enumerate(dataset):
        uf.print_numeric_progress(i, steps_per_epoch)
        depth_res, pose_res, traj = evaluate_batch(i, x, model)
        depth_result.append(depth_res)
        pose_result.append(pose_res)
        trajectory.append(traj)

    print("")
    depth_result = save_depth_result_and_get_df(depth_result, model_name)
    pose_result = save_pose_result_and_get_df(pose_result, model_name)
    save_trajectories(trajectory, model_name)

    depth_sample_inds = find_worst_depth_samples(depth_result, 5)
    print("worst depth sample indices\n", depth_sample_inds[0])
    pose_sample_inds = find_worst_pose_samples(pose_result, 5)
    print("worst pose sample indices\n", pose_sample_inds[0])
    worst_sample_inds = depth_sample_inds + pose_sample_inds

    pathname = op.join(opts.DATAPATH_EVL, model_name, 'debug_imgs')
    os.makedirs(pathname, exist_ok=True)

    for i, x in enumerate(dataset):
        uf.print_numeric_progress(i, steps_per_epoch)
        for sample_inds in worst_sample_inds:
            # sample_inds: df['frame', 'srcidx', metric or loss]
            save_worst_views(i, x, model, sample_inds, pathname)


def evaluate_batch(index, x, model):
    numsrc = opts.SNIPPET_LEN - 1

    stacked_image = x['image']
    intrinsic = x['intrinsic']
    depth_true = x['depth_gt']
    pose_true_mat = x['pose_gt']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)

    predictions = model(x['image'])
    disp_pred_ms = predictions['disp_ms']
    pose_pred = predictions['pose']
    depth_pred_ms = uf.safe_reciprocal_number_ms(disp_pred_ms)

    # evaluate depth from numpy arrays and take only 'abs_rel' metric
    depth_err, scale = compute_depth_error(depth_pred_ms[0].numpy()[0], depth_true.numpy()[0])
    smooth_loss = compute_smooth_loss(disp_pred_ms[0], target_image)

    pose_pred_mat = cp.pose_rvec2matr_batch(pose_pred)
    # pose error output: [batch, numsrc]
    trj_err, trj_len = compute_trajectory_error(pose_pred_mat, pose_true_mat, scale)
    rot_err = ef.calc_rotational_error_tensor(pose_pred_mat, pose_true_mat)

    # compute photometric loss: [batch, numsrc]
    photo_loss = compute_photo_loss(target_image, source_image, intrinsic, depth_pred_ms, pose_pred)

    depth_res = [index, smooth_loss, depth_err]
    # pose_res: [numsrc, -1]
    pose_res = np.stack([np.array([index] * 4), np.arange(numsrc), photo_loss.numpy().reshape(-1),
                         trj_err.numpy().reshape(-1), trj_len.numpy().reshape(-1),
                         rot_err.numpy().reshape(-1)], axis=1)

    # to collect trajectory
    trajectory = np.concatenate([np.array([index] * 4)[:, np.newaxis], np.arange(numsrc)[:, np.newaxis],
                                 pose_true_mat.numpy()[:, :, :3, 3].reshape((-1, 3)),
                                 pose_pred_mat.numpy()[:, :, :3, 3].reshape((-1, 3))*scale], axis=1)
    return depth_res, pose_res, trajectory


def compute_photo_loss(target_true, source_image, intrinsic, depth_pred_ms, pose_pred):
    # synthesize target image
    synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, depth_pred_ms, pose_pred)
    losses = []
    target_pred = synth_target_ms[0]
    # photometric loss: [batch, numsrc]
    loss = photometric_loss(target_pred, target_true)
    return loss


def compute_smooth_loss(disparity, target_image):
    # [batch]
    loss = smootheness_loss(disparity, target_image)
    # return scalar
    return loss.numpy()[0]


def compute_trajectory_error(pose_pred_mat, pose_true_mat, scale):
    """
    :param pose_pred_mat: predicted snippet pose matrices, [batch, numsrc, 4, 4]
    :param pose_true_mat: ground truth snippet pose matrices, [batch, numsrc, 4, 4]
    :param scale: scale for pose_pred to have real scale
    :return: trajectory error in meter [batch, numsrc]
    """
    xyz_pred = pose_pred_mat[:, :, :3, 3]
    xyz_true = pose_true_mat[:, :, :3, 3]
    # adjust the trajectory scaling due to ignorance of abolute scale
    # scale = tf.reduce_sum(xyz_true * xyz_pred, axis=2) / tf.reduce_sum(xyz_pred ** 2, axis=2)
    # scale = tf.expand_dims(scale, -1)
    traj_error = xyz_true - xyz_pred * tf.constant([[[scale]]])
    traj_error = tf.sqrt(tf.reduce_sum(traj_error ** 2, axis=2))
    traj_len = tf.sqrt(tf.reduce_sum(xyz_true ** 2, axis=2))
    return traj_error, traj_len


def compute_depth_error(depth_pred, depth_true):
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
    # return only abs rel
    return metrics[0], scaler


def save_depth_result_and_get_df(depth_result, model_name):
    depth_result = np.array(depth_result)
    depth_result = pd.DataFrame(data=depth_result, columns=['frame', 'smooth_loss', 'depth_err'])
    depth_result['frame'] = depth_result['frame'].astype(int)
    filename = op.join(opts.DATAPATH_EVL, model_name, 'debug_depth.csv')
    depth_result.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')
    return depth_result


def save_pose_result_and_get_df(pose_result, model_name):
    pose_result = np.concatenate(pose_result, axis=0)
    columns = ['frame', 'srcidx', 'photo_loss', 'trj_err', 'distance', 'rot_err']
    pose_result = pd.DataFrame(data=pose_result, columns=columns)
    pose_result['frame'] = pose_result['frame'].astype(int)
    pose_result['srcidx'] = pose_result['srcidx'].astype(int)
    filename = op.join(opts.DATAPATH_EVL, model_name, 'debug_pose.csv')
    pose_result.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')
    return pose_result


def save_trajectories(trajectory, model_name):
    trajectory = np.concatenate(trajectory, axis=0)
    trajectory = pd.DataFrame(data=trajectory, columns=['frame', 'srcidx', 'tx', 'ty', 'tz', 'px', 'py', 'pz'])
    trajectory['frame'] = trajectory['frame'].astype(int)
    trajectory['srcidx'] = trajectory['srcidx'].astype(int)
    filename = op.join(opts.DATAPATH_EVL, model_name, 'trajectory.csv')
    trajectory.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')


def find_worst_depth_samples(depth_result, num_samples):
    dfcols = list(depth_result)
    sample_inds = []
    for colname in ['depth_err']:
        sorted_result = depth_result[dfcols[:1] + [colname]].sort_values(by=[colname], ascending=False)
        sorted_result = sorted_result.reset_index(drop=True).head(num_samples)
        sorted_result['srcidx'] = 0
        sorted_result = sorted_result[['frame', 'srcidx', colname]]
        sample_inds.append(sorted_result)
    return sample_inds


def find_worst_pose_samples(pose_result, num_samples):
    dfcols = list(pose_result)
    sample_inds = []
    for colname in ['photo_loss', 'trj_err']:
        sorted_result = pose_result[dfcols[:2] + [colname]].sort_values(by=[colname], ascending=False)
        sorted_result = sorted_result.reset_index(drop=True).head(num_samples)
        sample_inds.append(sorted_result)
    return sample_inds


def save_worst_views(frame, x, model, sample_inds, save_path, scale=1):
    if frame not in sample_inds['frame'].tolist():
        return

    colname = list(sample_inds)[-1]
    indices = sample_inds.loc[sample_inds['frame'] == frame, :].index.tolist()

    stacked_image = x['image']
    intrinsic = x['intrinsic']
    depth_gt = x['depth_gt']
    pose_gt = x['pose_gt']
    pose_gt = cp.pose_matr2rvec_batch(pose_gt)
    depth_gt_ms = uf.multi_scale_depths(depth_gt, [1, 2, 4, 8])
    source_image, target_image = uf.split_into_source_and_target(stacked_image)

    predictions = model(x['image'])
    disp_pred_ms = predictions['disp_ms']
    pose_pred = predictions['pose']
    depth_pred_ms = uf.safe_reciprocal_number_ms(disp_pred_ms)

    depth_pred_ms = [depth*scale for depth in depth_pred_ms]

    synthesizer = SynthesizeMultiScale()
    synth_target_pred_ms = synthesizer(source_image, intrinsic, depth_pred_ms, pose_pred)
    synth_target_gt_ms = synthesizer(source_image, intrinsic, depth_gt_ms, pose_gt)

    for ind in indices:
        srcidx = sample_inds.loc[ind, 'srcidx']
        view_imgs = {"target": target_image, "synthesized": synth_target_pred_ms[0][0, srcidx],
                     "depth": depth_pred_ms[0][0, srcidx], "synth_by_gt": synth_target_gt_ms[0][0, srcidx]}
        view = uf.stack_titled_images(view_imgs)
        filename = op.join(save_path, f"{colname[:3]}_{frame:04d}_{srcidx}.png")
        print("save file:", filename)
        cv2.imwrite(filename, view)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    evaluate_for_debug('kitti_raw_test', 'vode1')

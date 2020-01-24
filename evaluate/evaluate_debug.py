import os
import os.path as op
import numpy as np
import pandas as pd
import cv2

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
import utils.util_funcs as uf
import utils.convert_pose as cp
import evaluate.eval_funcs as ef
from evaluate.evaluate_main import evaluate_depth
from model.synthesize_batch import synthesize_batch_multi_scale
from model.loss_and_metric import photometric_loss, smootheness_loss
from model.model_main import set_configs, create_model, try_load_weights


def evaluate_for_debug(data_dir_name, model_name):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, data_dir_name), batch_size=1).get_generator()
    depth_result = []
    pose_result = []
    steps_per_epoch = uf.count_steps(data_dir_name, 1)

    for i, x in enumerate(dataset):
        uf.print_numeric_progress(i, steps_per_epoch)
        depth_res, pose_res = evaluate_batch(i, x, model)
        depth_result.append(depth_res)
        pose_result.append(pose_res)

    print("")
    depth_result = save_depth_result_and_get_df(depth_result, model_name)
    print("depth_result\n", depth_result.head(10))
    pose_result = save_pose_result_and_get_df(pose_result, model_name)
    print("pose_result\n", pose_result.head(10))

    depth_sample_inds = find_worst_depth_samples(depth_result, 5)
    print("depth sample indices\n", depth_sample_inds[0])
    pose_sample_inds = find_worst_pose_samples(pose_result, 5)
    print("pose sample indices\n", pose_sample_inds[0])

    pathname = op.join(opts.DATAPATH_EVL, model_name, 'debug_imgs')
    os.makedirs(pathname, exist_ok=True)

    for i, x in enumerate(dataset):
        uf.print_numeric_progress(i, steps_per_epoch)
        for sample_inds in depth_sample_inds:
            save_worst_depth_view(i, x, model, sample_inds, pathname)
        for sample_inds in pose_sample_inds:
            save_worst_pose_view(i, x, model, sample_inds, pathname)


def evaluate_batch(index, x, model):
    num_src = opts.SNIPPET_LEN - 1

    stacked_image = x['image']
    intrinsic = x['intrinsic']
    depth_true = x['depth_gt']
    pose_true_mat = x['pose_gt']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)

    predictions = model(x['image'])
    disp_pred_ms = predictions['disp_ms']
    pose_pred = predictions['pose']
    depth_pred_ms = uf.disp_to_depth_tensor(disp_pred_ms)

    # evaluate depth from numpy arrays and take only 'abs_rel' metric
    depth_err = evaluate_depth(depth_pred_ms[0].numpy()[0], depth_true.numpy()[0])
    depth_err = depth_err[0]
    smooth_loss = compute_smooth_loss(disp_pred_ms[0], target_image)

    snp_res = [index, smooth_loss, depth_err]

    # evaluate poses
    pose_pred_mat = cp.pose_rvec2matr_batch(pose_pred)
    # pose error output: [batch, num_src]
    trj_err = ef.calc_trajectory_error_tensor(pose_pred_mat, pose_true_mat)
    rot_err = ef.calc_rotational_error_tensor(pose_pred_mat, pose_true_mat)
    # compute photometric loss: [batch, num_src]
    photo_loss = compute_photo_loss(target_image, source_image, intrinsic, depth_pred_ms, pose_pred)

    # src_res: [num_src, -1]
    src_res = np.stack([np.array([index] * 4), np.arange(num_src), photo_loss.numpy().reshape(-1),
                        trj_err.numpy().reshape(-1), rot_err.numpy().reshape(-1)], axis=1)
    return snp_res, src_res


def compute_photo_loss(target_true, source_image, intrinsic, depth_pred_ms, pose_pred):
    # synthesize target image
    synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, depth_pred_ms, pose_pred)
    losses = []
    target_pred = synth_target_ms[0]
    # photometric loss: [batch, num_src]
    loss = photometric_loss(target_pred, target_true)
    return loss


def compute_smooth_loss(disparity, target_image):
    # [batch]
    loss = smootheness_loss(disparity, target_image)
    # return scalar
    return loss.numpy()[0]


def save_depth_result_and_get_df(depth_result, model_name):
    depth_result = np.array(depth_result)
    depth_result = pd.DataFrame(data=depth_result, columns=['frame', 'smooth_loss', 'depth_err'])
    depth_result['frame'] = depth_result['frame'].astype(int)
    filename = op.join(opts.DATAPATH_EVL, model_name, 'debug_depth.csv')
    depth_result.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')
    return depth_result


def save_pose_result_and_get_df(pose_result, model_name):
    pose_result = np.concatenate(pose_result, axis=0)
    columns = ['frame', 'source', 'photo_loss', 'trj_err', 'rot_err']
    pose_result = pd.DataFrame(data=pose_result, columns=columns)
    pose_result['frame'] = pose_result['frame'].astype(int)
    pose_result['source'] = pose_result['source'].astype(int)
    filename = op.join(opts.DATAPATH_EVL, model_name, 'debug_pose.csv')
    pose_result.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')
    return pose_result


def find_worst_depth_samples(depth_result, num_samples):
    sample_inds = []
    for colname in ['smooth_loss', 'depth_err']:
        sorted_result = depth_result[['frame'] + [colname]].sort_values(by=[colname], ascending=False)
        sorted_result = sorted_result.reset_index(drop=True).head(num_samples)
        sample_inds.append(sorted_result)
    return sample_inds


def find_worst_pose_samples(pose_result, num_samples):
    sample_inds = []
    for colname in ['photo_loss', 'trj_err', 'rot_err']:
        sorted_result = pose_result[['frame', 'source'] + [colname]].sort_values(by=[colname], ascending=False)
        sorted_result = sorted_result.reset_index(drop=True).head(num_samples)
        sample_inds.append(sorted_result)
    return sample_inds


def save_worst_pose_view(frame, x, model, sample_inds, save_path):
    if frame not in sample_inds['frame'].tolist():
        return

    colname = list(sample_inds)[-1]
    indices = sample_inds.loc[sample_inds['frame'] == frame, :].index.tolist()

    stacked_image = x['image']
    intrinsic = x['intrinsic']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)

    predictions = model(x['image'])
    disp_pred_ms = predictions['disp_ms']
    pose_pred = predictions['pose']
    depth_pred_ms = uf.disp_to_depth_tensor(disp_pred_ms)

    synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, depth_pred_ms, pose_pred)

    for ind in indices:
        srcidx = sample_inds.loc[ind, 'source']
        view = uf.make_view(target_image, synth_target_ms[0], depth_pred_ms[0],
                            source_image, batidx=0, srcidx=srcidx, verbose=False)
        filename = op.join(save_path, f"{colname[:3]}_{frame:04d}_{srcidx}.png")
        print("save file:", filename)
        cv2.imwrite(filename, view)


def save_worst_depth_view(frame, x, model, sample_inds, save_path):
    if frame not in sample_inds['frame'].tolist():
        return

    colname = list(sample_inds)[-1]

    stacked_image = x['image']
    intrinsic = x['intrinsic']
    source_image, target_image = uf.split_into_source_and_target(stacked_image)

    predictions = model(x['image'])
    disp_pred_ms = predictions['disp_ms']
    pose_pred = predictions['pose']
    depth_pred_ms = uf.disp_to_depth_tensor(disp_pred_ms)

    synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, depth_pred_ms, pose_pred)

    view = uf.make_view(target_image, synth_target_ms[0], depth_pred_ms[0],
                        source_image, batidx=0, srcidx=0, verbose=False)
    filename = op.join(save_path, f"{colname[:3]}_{frame:04d}.png")
    print("save file:", filename)
    cv2.imwrite(filename, view)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    evaluate_for_debug('kitti_raw_test', 'vode1')

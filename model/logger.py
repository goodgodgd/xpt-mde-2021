import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import importlib
import shutil

from config import opts
import utils.util_funcs as uf
import model.loss_and_metric.losses as lm
from model.synthesize.synthesize_base import SynthesizeMultiScale
import utils.convert_pose as cp


def save_log(epoch, results_train, results_val, depth_train, depth_val):
    """
    :param epoch: current epoch
    :param results_train: (loss, metric_trj, metric_rot, (losses from various loss types))
    :param results_val: (loss, metric_trj, metric_rot, (losses from various loss types))
    """
    results = save_results(epoch, results_train[:3], results_val[:3], ["loss", "trj_err", "rot_err"], "history.txt")
    _ = save_results(epoch, results_train[3:], results_val[3:], list(opts.LOSS_WEIGHTS.keys()), "losses.txt")
    save_depths(depth_train, depth_val, "depths.txt")
    draw_and_save_plot(results, "history.png")


def save_results(epoch, results_train, results_val, columns, filename):
    results = np.concatenate([[epoch], results_train, results_val], axis=0)
    results = np.expand_dims(results, 0)

    train_columns = ["train_" + col for col in columns]
    val_columns = ["val_" + col for col in columns]
    total_columns = ["epoch"] + train_columns + val_columns
    # create single row dataframe
    results = pd.DataFrame(data=results, columns=total_columns)
    results["|"] = "|"
    total_columns = ["epoch"] + train_columns + ["|"] + val_columns
    results = results[total_columns]
    results['epoch'] = results['epoch'].astype(int)

    filepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, filename)
    # if the file existed, append new data to it
    if op.isfile(filepath):
        existing = pd.read_csv(filepath, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        results = pd.concat([existing, results], axis=0, ignore_index=True)
        results = results.drop_duplicates(subset='epoch', keep='last')
        results = results.sort_values(by=['epoch'])
    # write to a file
    results.to_csv(filepath, encoding='utf-8', index=False, float_format='%.4f')
    return results


def save_depths(depth_train, depth_val, filename):
    """
    depth_xxx: mean depths from a epoch, true depths in row 0, predicted depths in row 1
    """
    depths = np.concatenate([depth_train, [[-1], [-2]], depth_val], axis=1)
    filepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, filename)
    # if the file existed, append only row 1 to it
    if op.isfile(filepath):
        existing = np.loadtxt(filepath)
        depths = np.concatenate([existing, [depths[1]]], axis=0)
    # write to a file
    np.savetxt(filepath, depths, fmt="%7.4f")


def draw_and_save_plot(results, filename):
    # plot graphs of loss and metrics
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(7, 7)
    for i, ax, colname, title in zip(range(3), axes, ['loss', 'trj_err', 'rot_err'], ['Loss', 'Trajectory Error', 'Rotation Error']):
        ax.plot(results['epoch'], results['train_' + colname], label='train_' + colname)
        ax.plot(results['epoch'], results['val_' + colname], label='val_' + colname)
        ax.set_xlabel('epoch')
        ax.set_ylabel(colname)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    # save graph as a file
    filepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, filename)
    fig.savefig(filepath, dpi=100)
    plt.close("all")


def save_reconstruction_samples(model, dataset, epoch):
    views = make_reconstructed_views(model, dataset)
    savepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, 'reconimg')
    if not op.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    for i, view in enumerate(views):
        filename = op.join(savepath, f"ep{epoch:03d}_{i:02d}.png")
        cv2.imwrite(filename, view)


def make_reconstructed_views(model, dataset):
    recon_views = []
    next_idx = 0
    stride = 10
    stereo_loss = lm.StereoDepthLoss("L1")
    total_loss = lm.TotalLoss()

    for i, features in enumerate(dataset):
        if i < next_idx:
            continue
        if i // stride > 5:
            stride *= 10
        next_idx += stride

        predictions = model(features)
        augm_data = total_loss.augment_data(features, predictions)
        if opts.STEREO:
            augm_data_rig = total_loss.augment_data(features, predictions, "_R")
            augm_data.update(augm_data_rig)

        synth_target_ms = SynthesizeMultiScale()(src_img_stacked=augm_data['source'],
                                                 intrinsic=features['intrinsic'],
                                                 pred_depth_ms=augm_data['depth_ms'],
                                                 pred_pose=predictions['pose'])

        scaleidx, batchidx, srcidx = 0, 0, 0
        target_depth = augm_data["depth_ms"][0][batchidx]
        target_depth = tf.clip_by_value(target_depth, 0., 20.) / 10. - 1.
        source_time = augm_data["source"][batchidx, srcidx*opts.IM_HEIGHT:(srcidx + 1)*opts.IM_HEIGHT]
        view_imgs = [augm_data["target"][0],
                     target_depth,
                     source_time,
                     synth_target_ms[scaleidx][batchidx, srcidx]]
        view_names = ["left_target", "target_depth", f"source_{srcidx}", f"synthesized_from_src{srcidx}"]

        if opts.STEREO:
            depths_ms = []
            for dep in augm_data["depth_ms"]:
                depth = dep + 0.
                depths_ms.append(depth)

            batch_loss = stereo_loss(features, predictions, augm_data)

            loss_left, synth_left_ms = \
                stereo_loss.stereo_synthesize_loss(source_img=augm_data["target_R"],
                                                   target_ms=augm_data["target_ms"],
                                                   target_depth_ms=depths_ms,
                                                   pose_t2s=tf.linalg.inv(features["stereo_T_LR"]),
                                                   intrinsic=features["intrinsic"])

            loss_again = []
            for k, (synth_img_sc, target_img_sc) in enumerate(zip(synth_left_ms, augm_data["target_ms"])):
                loss = lm.photometric_loss_l1(synth_img_sc, target_img_sc)
                loss_again.append(loss)
                synt_target_gray = tf.reduce_mean(synth_img_sc, axis=-1, keepdims=True)
                error_mask = tf.equal(synt_target_gray, 0)
                print("mask invalid count", k, tf.size(error_mask), tf.reduce_sum(tf.cast(error_mask, tf.int32)).numpy())

            print("batch loss", batch_loss)
            print("synth size", tf.size(synth_left_ms[scaleidx]), synth_left_ms[scaleidx].get_shape().as_list())
            print("synth zero count", tf.reduce_sum(tf.cast(tf.math.equal(synth_left_ms[scaleidx], 0.), tf.int32)).numpy())
            print("loss left", loss_left[0], "\n", loss_left[1])
            print("loss again", loss_again[0], "\n", loss_again[1])

            # print("depths:", depths_ms[0][0, 50:100:10, 100:300:30, 0])
            # print("pose:", tf.linalg.inv(features["stereo_T_LR"])[0])
            # print("intrinsic:", features["intrinsic"][0])
            view_imgs.append(augm_data["target_R"][batchidx])
            view_imgs.append(synth_left_ms[scaleidx][batchidx, srcidx])
            view_names.append("right_source")
            view_names.append("synthesized_from_right")

        view1 = uf.make_view2(view_imgs, view_names)
        recon_views.append(view1)

    return recon_views


def save_loss_scales(model, dataset, steps, is_stereo):
    if opts.LOG_LOSS:
        print("\n===== save_loss_scales")
        losses = collect_losses(model, dataset, steps, is_stereo)
        save_loss_to_file(losses)


def collect_losses(model, dataset, steps_per_epoch, is_stereo):
    results = {"L1": [], "SSIM": [], "smoothe": [], "stereo": []}
    total_loss = lm.TotalLoss()
    calc_photo_loss_l1 = lm.PhotometricLossMultiScale("L1")
    calc_photo_loss_ssim = lm.PhotometricLossMultiScale("SSIM")
    calc_smootheness_loss = lm.SmoothenessLossMultiScale()
    calc_stereo_loss = lm.StereoDepthLoss("L1")
    stride = steps_per_epoch // 20

    for step, features in enumerate(dataset):
        if step % stride > 0:
            continue

        preds = model(features)
        augm_data = total_loss.augment_data(features, preds)
        if is_stereo:
            augm_data_rig = total_loss.augment_data(features, preds, "_R")
            augm_data.update(augm_data_rig)

        photo_l1 = calc_photo_loss_l1(features, preds, augm_data)
        photo_ssim = calc_photo_loss_ssim(features, preds, augm_data)
        smoothe = calc_smootheness_loss(features, preds, augm_data)
        stereo = calc_stereo_loss(features, preds, augm_data)

        results["L1"].append(photo_l1)
        results["SSIM"].append(photo_ssim)
        results["smoothe"].append(smoothe)
        results["stereo"].append(stereo)
        uf.print_progress_status(f"step: {step} / {steps_per_epoch}")

    print("")
    results = {key: tf.concat(res, 0).numpy() for key, res in results.items()}
    return results


def save_loss_to_file(losses):
    with open(op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, "loss_scale.txt"), "a") as f:
        for key, loss in losses.items():
            f.write(f"> loss type={key}, shape={loss.shape}\n")
            f.write(f"\tloss min={loss.min():1.4f}, max={loss.max():1.4f}, mean={loss.mean():1.4f}, median={np.median(loss):1.4f}\n")
            f.write(f"\tloss quantile={np.quantile(loss, np.arange(0, 1, 0.1))}\n")
        f.write("\n\n")
        print("loss_scale.txt written !!")


def copy_or_check_same():
    saved_conf_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, "saved_config.py")
    if not op.isfile(saved_conf_path):
        shutil.copyfile(op.join(opts.PROJECT_ROOT, "config.py"), saved_conf_path)
        return

    import sys
    if opts.DATAPATH_CKP not in sys.path:
        sys.path.append(opts.DATAPATH_CKP)
    print(sys.path)
    class_path = opts.CKPT_NAME + ".saved_config" + ".VodeOptions"
    module_name, class_name = class_path.rsplit('.', 1)
    module_obj = importlib.import_module(module_name)
    SavedOptions = getattr(module_obj, class_name)
    from config import VodeOptions as CurrOptions

    saved_opts = {attr: SavedOptions.__dict__[attr] for attr in SavedOptions.__dict__ if
                  not callable(getattr(SavedOptions, attr)) and not attr.startswith('__')}
    curr_opts = {attr: CurrOptions.__dict__[attr] for attr in CurrOptions.__dict__ if
                 not callable(getattr(CurrOptions, attr)) and not attr.startswith('__')}

    dont_care_opts = ["BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "ENABLE_SHAPE_DECOR",
                      "CKPT_NAME", "LOG_LOSS", "PROJECT_ROOT"]

    for key, saved_val in saved_opts.items():
        if key in dont_care_opts:
            continue
        curr_val = curr_opts[key]
        assert saved_val == curr_val, f"key: {key}, {curr_val} != {saved_val}"

    print("!! config comparison passed !!")

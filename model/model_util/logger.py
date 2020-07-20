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


def save_log(epoch, results_train, results_val):
    """
    :param epoch: current epoch
    :param results_train: dict of losses, metrics and depths from training data
    :param results_val: dict of losses, metrics and depths from validation data
    """
    summary = save_results(epoch, results_train, results_val, ["loss", "trjerr", "roterr"], "history.csv")
    other_cols = [colname for colname in results_train.keys() if colname not in ["loss", "trjerr", "roterr"]]
    _ = save_results(epoch, results_train, results_val, other_cols, "mean_result.csv")

    save_scales(epoch, results_train, results_val, "scales.txt")
    draw_and_save_plot(summary, "history.png")


def save_results(epoch, results_train, results_val, columns, filename):
    train_result = results_train.mean(axis=0).to_dict()
    val_result = results_val.mean(axis=0).to_dict()
    epoch_result = {"epoch": epoch}
    for colname in columns:
        epoch_result["t_" + colname] = train_result[colname]
    epoch_result["|"] = "|"
    for colname in columns:
        epoch_result["v_" + colname] = val_result[colname]

    filepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, filename)
    # if the file existed, append new data to it
    if op.isfile(filepath):
        existing = pd.read_csv(filepath, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        results = existing.append(epoch_result, ignore_index=True)
        results = results.drop_duplicates(subset='epoch', keep='last')
        results = results.sort_values(by=['epoch'])
    else:
        results = pd.DataFrame([epoch_result])

    # reorder columns
    loss_cols = list(opts.LOSS_WEIGHTS.keys())
    other_cols = [col for col in columns if col not in loss_cols]
    split_cols = loss_cols + other_cols
    train_cols = ["t_" + col for col in split_cols]
    val_cols = ["v_" + col for col in split_cols]
    total_cols = ["epoch"] + train_cols + ["|"] + val_cols
    print("total cols", total_cols)
    print("result", list(results))
    results = results.loc[:, total_cols]

    # write to a file
    results['epoch'] = results['epoch'].astype(int)
    results.to_csv(filepath, encoding='utf-8', index=False, float_format='%.4f')
    print(f"write {filename}\n", results.tail())
    return results


def draw_and_save_plot(results, filename):
    # plot graphs of loss and metrics
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(7, 7)
    for i, ax, colname, title in zip(range(3), axes, ['loss', 'trjerr', 'roterr'], ['Loss', 'Trajectory Error', 'Rotation Error']):
        ax.plot(results['epoch'], results['t_' + colname], label='train_' + colname)
        ax.plot(results['epoch'], results['v_' + colname], label='val_' + colname)
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
    scaleidx, batchidx, srcidx = 0, 0, 0

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

        synth_target_ms = SynthesizeMultiScale()(src_img_stacked=augm_data["source"],
                                                 intrinsic=features["intrinsic"],
                                                 pred_depth_ms=predictions["depth_ms"],
                                                 pred_pose=predictions["pose"])

        target_depth = predictions["depth_ms"][0][batchidx]
        target_depth = tf.clip_by_value(target_depth, 0., 20.) / 10. - 1.
        time_source = augm_data["source"][batchidx, srcidx*opts.IM_HEIGHT:(srcidx + 1)*opts.IM_HEIGHT]
        view_imgs = {"left_target": augm_data["target"][0],
                     "target_depth": target_depth,
                     f"source_{srcidx}": time_source,
                     f"synthesized_from_src{srcidx}": synth_target_ms[scaleidx][batchidx, srcidx]
                     }
        view_imgs["time_diff"] = tf.abs(view_imgs["left_target"] - view_imgs[f"synthesized_from_src{srcidx}"])

        if opts.STEREO:
            loss_left, synth_left_ms = \
                stereo_loss.stereo_synthesize_loss(source_img=augm_data["target_R"],
                                                   target_ms=augm_data["target_ms"],
                                                   target_depth_ms=predictions["depth_ms"],
                                                   pose_t2s=tf.linalg.inv(features["stereo_T_LR"]),
                                                   intrinsic=features["intrinsic"])

            # print("stereo synth size", tf.size(synth_left_ms[scaleidx]).numpy())
            # zeromask = tf.cast(tf.math.equal(synth_left_ms[scaleidx], 0.), tf.int32)
            # print("stereo synth zero count", tf.reduce_sum(zeromask).numpy())
            print(f"saving synthesized image {i}, stereo loss R2L:", tf.squeeze(loss_left[0]).numpy())
            view_imgs["right_source"] = augm_data["target_R"][batchidx]
            view_imgs["synthesized_from_right"] = synth_left_ms[scaleidx][batchidx, srcidx]
            view_imgs["stereo_diff"] = tf.abs(view_imgs["left_target"] - view_imgs["synthesized_from_right"])

        view1 = uf.stack_titled_images(view_imgs)
        recon_views.append(view1)

    return recon_views


def save_scales(epoch, results_train, results_val, filename):
    filepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, filename)
    results_train = results_train.rename(columns={col: "t_" + col for col in list(results_train)})
    results_val = results_val.rename(columns={col: "v_" + col for col in list(results_val)})
    results = pd.concat([results_train, results_val], axis=1)
    results = results.quantile([0, 0.25, 0.5, 0.75, 1.], axis=0)
    results["|"] = "|"
    results = results[list(results_train) + ["|"] + list(results_val)]

    with open(filepath, "a") as f:
        f.write(f"===== epoch: {epoch}\n")
        f.write(f"{results.to_csv(sep=' ', index=False, float_format='%.4f')}\n\n")
        print(f"{filename} written !!")


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
                      "CKPT_NAME", "LOG_LOSS", "PROJECT_ROOT", "DATASET", "TRAIN_MODE", "LOSS_WEIGHTS"]

    for key, saved_val in saved_opts.items():
        if key in dont_care_opts:
            continue
        curr_val = curr_opts[key]
        assert saved_val == curr_val, f"key: {key}, {curr_val} != {saved_val}"

    print("!! config comparison passed !!")

import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import importlib
import shutil

from config import opts
import utils.util_funcs as uf
import model.loss_and_metric.losses as lm


def save_log(epoch, results_train, results_val):
    """
    :param epoch: current epoch
    :param results_train: dict of losses, metrics and depths from training data
    :param results_val: dict of losses, metrics and depths from validation data
    """
    summ_cols = ["loss", "trjerr", "roterr"]
    summary = save_results(epoch, results_train, results_val, summ_cols, "history.csv")
    other_cols = [colname for colname in results_train.keys() if colname not in summ_cols]
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
    all_loss_cols = list(opts.LOSS_WEIGHTS.keys())
    loss_cols = [col for col in columns if col in all_loss_cols]
    other_cols = [col for col in columns if col not in all_loss_cols]
    split_cols = loss_cols + other_cols
    train_cols = ["t_" + col for col in split_cols]
    val_cols = ["v_" + col for col in split_cols]
    total_cols = ["epoch"] + train_cols + ["|"] + val_cols
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


def save_reconstruction_samples(model, dataset, total_steps, epoch):
    views = make_reconstructed_views(model, dataset, total_steps)
    savepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, 'reconimg')
    if not op.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    for i, view in enumerate(views):
        filename = op.join(savepath, f"ep{epoch:03d}_{i:02d}.png")
        cv2.imwrite(filename, view)


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


def make_reconstructed_views(model, dataset, total_steps):
    recon_views = []
    # 7 file are in a row in file explorer
    stride = min(total_steps, 400) // 7
    max_steps = stride * 7
    total_loss = lm.TotalLoss()
    scaleidx, batchidx, srcidx = 0, 0, 0

    for i, features in enumerate(dataset):
        if i % stride != 1:
            continue
        if i > max_steps:
            break

        # predict by model
        predictions = model(features)

        # create intermediate data
        augm_data = total_loss.augment_data(features, predictions)
        if opts.STEREO:
            augm_data_rig = total_loss.augment_data(features, predictions, "_R")
            augm_data.update(augm_data_rig)
            augm_data_stereo = total_loss.synethesize_stereo(features, predictions, augm_data)
            augm_data.update(augm_data_stereo)

        target_depth = predictions["depth_ms"][0][batchidx]
        target_depth = tf.clip_by_value(target_depth, 0., 20.) / 10. - 1.
        time_source = augm_data["source"][batchidx, srcidx*opts.IM_HEIGHT:(srcidx + 1)*opts.IM_HEIGHT]
        view_imgs = {"left_target": augm_data["target"][0],
                     "target_depth": target_depth,
                     f"source_{srcidx}": time_source,
                     f"synthesized_from_src{srcidx}": augm_data["synth_target_ms"][scaleidx][batchidx, srcidx]
                     }
        # view_imgs["time_diff"] = tf.abs(view_imgs["left_target"] - view_imgs[f"synthesized_from_src{srcidx}"])

        if "flow_ms" in predictions:
            view_imgs["synthesized_by_flow"] = augm_data["warped_target_ms"][scaleidx][batchidx, srcidx]

        if opts.STEREO:
            view_imgs["right_source"] = augm_data["target_R"][batchidx]
            view_imgs["synthesized_from_right"] = augm_data["stereo_synth_ms"][scaleidx][batchidx, srcidx]
            # view_imgs["stereo_diff"] = tf.abs(view_imgs["left_target"] - view_imgs["synthesized_from_right"])

        view1 = uf.stack_titled_images(view_imgs)
        recon_views.append(view1)
    return recon_views


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

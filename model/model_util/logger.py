import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import importlib
import shutil
import copy
import json

from config import opts
import utils.util_funcs as uf
import model.loss_and_metric.losses as lm

RENAMER = {"trjabs": "TEA", "trjrel": "TER", "roterr": "RE", "deprel": "DE", "depth": "dp",
           "SSIM": "SS", "smoothe": "sm", "pose": "ps", "stereo": "st", "flow": "fl",
           "stereoPose": "stps", "_reg": "Rg", "_R": "R"}
TRAIN_PREFIX = ":"
VALID_PREFIX = "!"
RECON_SAMPLES = 14


def save_log(epoch, dataset_name, results_train, results_val):
    """
    :param epoch:
    :param dataset_name:
    :param results_train: dict of losses, metrics and depths from training data
    :param results_val: dict of losses, metrics and depths from validation data
    """
    summ_cols = ["loss", "trjabs", "trjrel", "roterr", "deprel"]
    summary = save_results(epoch, dataset_name, results_train, results_val, summ_cols, "history.csv")
    other_cols = [colname for colname in results_train.keys() if colname not in summ_cols]
    _ = save_results(epoch, dataset_name, results_train, results_val, other_cols, "mean_result.csv")

    save_scales(epoch, results_train, results_val, "scales.txt")
    draw_and_save_plot(summary, "history.png")


def save_results(epoch, dataset_name, results_train, results_val, columns, filename):
    """
    저장된 csv 파일에서 하나의 column 너비는 가급적 6 글자가 되도록 맞춘다.
    예시:
        epoch,dataset,:loss ,:TE   ,:RE   ,  |   ,!loss ,!TE   ,!RE
            0,kitti_r,1.9476,1.3867,0.0130,  |   ,1.1700,0.2056,0.0065
            1,kitti_r,1.8911,1.3442,0.0129,  |   ,1.1845,0.2120,0.0051

    - column 이름에서 ':'는 training 결과를 말하고 '!'는 validation 결과를 뜻한다.
    - 6글자에 맞추기 위해 단어들을 줄여서 썼는데 약자들은 상단의 RENAMER나
      checkpts의 how-to-read-columns.txt 에서도 확인할 수 있다.
    - smootheness loss나 regularization loss는 크기가 작아서 1000을 곱해서 저장한다.
    """
    train_result = results_train.mean(axis=0).to_dict()
    val_result = results_val.mean(axis=0).to_dict()

    epoch_result = {"epoch": f"{epoch:>5}", "dataset": f"{dataset_name[:7]:<7}"}
    for colname in columns:
        if colname in train_result:
            epoch_result[TRAIN_PREFIX + colname] = train_result[colname]
    seperator = "  |   "
    epoch_result[seperator] = seperator
    for colname in columns:
        if colname in val_result:
            epoch_result[VALID_PREFIX + colname] = val_result[colname]
    epoch_result = to_fixed_width_column(epoch_result)

    # save "how-to-read-columns.json"
    renamerfile = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, "how-to-read-columns.txt")
    if not op.isfile(renamerfile):
        with open(renamerfile, "w") as fw:
            json.dump(RENAMER, fw, separators=(',\n', ': '))
            fw.write("\n\nSmootheness loss and flow reguluarization loss are scaled up to x1000\n")

    filepath = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, filename)
    # if the file existed, append new data to it
    if op.isfile(filepath):
        existing = pd.read_csv(filepath, encoding='utf-8', converters={'epoch': lambda c: f"{int(c):<5}"})
        results = existing.append(epoch_result, ignore_index=True)
        results = results.drop_duplicates(subset='epoch', keep='last')
        results = results.fillna(0.0)
        # reorder columns
        train_cols = [col for col in list(results) if col.startswith(TRAIN_PREFIX)]
        val_cols = [col for col in list(results) if col.startswith(VALID_PREFIX)]
        columns = ["epoch", "dataset"] + train_cols + [seperator] + val_cols
        results = results.loc[:, columns]
        # reorder rows
        results["epoch"] = results["epoch"].map(lambda x: int(x))
        results = results.sort_values(by=['epoch'])
        results["epoch"] = results["epoch"].map(lambda x: f"{x:>5}")
    else:
        results = pd.DataFrame([epoch_result])

    # write to a file
    results.to_csv(filepath, encoding='utf-8', index=False, float_format='%.4f')
    print(f"write {filename}\n", results.tail())
    return results


def to_fixed_width_column(srcdict):
    # scale up small values
    middict = copy.deepcopy(srcdict)
    for key, val in srcdict.items():
        if "smooth" in key.lower() or "reg" in key.lower():
            middict[key] = val * 1000.

    # rename keys to shorter strings
    dstdict = dict()
    for key, val in middict.items():
        newkey = copy.deepcopy(key)
        for old, new in RENAMER.items():
            if old in newkey:
                newkey = newkey.replace(old, new)

        if (newkey != "epoch") and (newkey != "dataset"):
            newkey = f"{newkey[:6]:<6}"
        dstdict[newkey] = val
    return dstdict


def draw_and_save_plot(results, filename):
    # plot graphs of loss and metrics
    sel_columns = ['loss ', 'TEA  ', 'TER  ', 'RE   ']
    col_titles = ['Loss', 'Traj abs. Error', 'Traj rel. Error', 'Rotation Error']
    numcols = len(col_titles)
    fig, axes = plt.subplots(numcols, 1)
    fig.set_size_inches(numcols*2, 7)
    for i, ax, colname, title in zip(range(numcols), axes, sel_columns, col_titles):
        if TRAIN_PREFIX + colname in results:
            ax.plot(results['epoch'].astype(int), results[TRAIN_PREFIX + colname], label='train_' + colname)
            ax.plot(results['epoch'].astype(int), results[VALID_PREFIX + colname], label='val_' + colname)
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
    stride = min(total_steps, RECON_SAMPLES*50) // RECON_SAMPLES
    max_steps = stride * RECON_SAMPLES
    total_loss = lm.TotalLoss()
    scaleidx, batchidx, srcidx = 0, 0, 0

    for i, features in enumerate(dataset):
        if i % stride != 1:
            continue
        if i > max_steps:
            break

        # predict by model
        predictions = model(features)
        view1 = stack_reconstruction_images(total_loss, features, predictions, (scaleidx, batchidx, srcidx))
        recon_views.append(view1)
    return recon_views


def stack_reconstruction_images(total_loss, features, predictions, indices):
    scaleidx, batchidx, srcidx = indices
    # create intermediate data
    augm_data = total_loss.append_data(features, predictions)
    if opts.STEREO and ("image_R" in features):
        augm_data_rig = total_loss.append_data(features, predictions, "_R")
        augm_data.update(augm_data_rig)
        augm_data_stereo = total_loss.synethesize_stereo(features, predictions, augm_data)
        augm_data.update(augm_data_stereo)

    view_imgs = {"left_target": augm_data["target"][0]}

    if "depth_ms" in predictions:
        target_depth = predictions["depth_ms"][0][batchidx]
        target_depth = tf.clip_by_value(target_depth, 0., 20.) / 10. - 1.
        view_imgs["target_depth"] = target_depth

    view_imgs[f"source_{srcidx}"] = augm_data["source"][batchidx, srcidx]

    if "synth_target_ms" in augm_data:
        view_imgs[f"synthesized_from_src{srcidx}"] = augm_data["synth_target_ms"][scaleidx][batchidx, srcidx]
        # view_imgs["time_diff"] = tf.abs(view_imgs["left_target"] - view_imgs[f"synthesized_from_src{srcidx}"])

    if "warped_target_ms" in augm_data:
        view_imgs["synthesized_by_flow"] = augm_data["warped_target_ms"][scaleidx][batchidx, srcidx]

    if opts.STEREO and ("stereo_synth_ms" in augm_data):
        view_imgs["right_source"] = augm_data["target_R"][batchidx]
        view_imgs["synthesized_from_right"] = augm_data["stereo_synth_ms"][scaleidx][batchidx, srcidx]
        # view_imgs["stereo_diff"] = tf.abs(view_imgs["left_target"] - view_imgs["synthesized_from_right"])

    view1 = uf.stack_titled_images(view_imgs)
    return view1


def copy_or_check_same():
    saved_conf_path = op.join(opts.DATAPATH_CKP, opts.CKPT_NAME, "saved_config.py")
    if not op.isfile(saved_conf_path):
        shutil.copyfile(op.join(opts.PROJECT_ROOT, "config.py"), saved_conf_path)
        return

    import sys
    if opts.DATAPATH_CKP not in sys.path:
        sys.path.append(opts.DATAPATH_CKP)
    print(sys.path)
    class_path = opts.CKPT_NAME + ".saved_config" + ".FixedOptions"
    module_name, class_name = class_path.rsplit('.', 1)
    module_obj = importlib.import_module(module_name)
    SavedOptions = getattr(module_obj, class_name)
    from config import FixedOptions as CurrOptions

    saved_opts = {attr: SavedOptions.__dict__[attr] for attr in SavedOptions.__dict__ if
                  not callable(getattr(SavedOptions, attr)) and not attr.startswith('__')}
    curr_opts = {attr: CurrOptions.__dict__[attr] for attr in CurrOptions.__dict__ if
                 not callable(getattr(CurrOptions, attr)) and not attr.startswith('__')}

    for key, saved_val in saved_opts.items():
        curr_val = curr_opts[key]
        assert saved_val == curr_val, f"key: {key}, {curr_val} != {saved_val}"

    print("!! config comparison passed !!")

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from config import opts
import utils.util_funcs as uf
from model.synthesize.synthesize_base import SynthesizeMultiScale


def save_log(epoch, results_train, results_val, ckpt_name):
    """
    :param epoch: current epoch
    :param results_train: (loss, metric_trj, metric_rot) from train data
    :param results_val: (loss, metric_trj, metric_rot) from validation data
    :param ckpt_name: model directory name
    """
    results = np.concatenate([[epoch], results_train, results_val], axis=0)
    results = np.expand_dims(results, 0)
    columns = ['epoch', 'train_loss', 'train_metric_trj', 'train_metric_rot', 'val_loss', 'val_metric_trj', 'val_metric_rot']
    results = pd.DataFrame(data=results, columns=columns)
    results['epoch'] = results['epoch'].astype(int)

    filename = op.join(opts.DATAPATH_CKP, ckpt_name, 'history.txt')
    # if the file existed, append new data to it
    if op.isfile(filename):
        existing = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        results = pd.concat([existing, results], axis=0, ignore_index=True)
        results = results.drop_duplicates(subset='epoch', keep='last')
        results = results.sort_values(by=['epoch'])
    # write to a file
    results.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')

    # plot graphs of loss and metrics
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(7, 7)
    for i, ax, colname, title in zip(range(3), axes, ['loss', 'metric_trj', 'metric_rot'], ['Loss', 'Trajectory Error', 'Rotation Error']):
        ax.plot(results['epoch'], results['train_' + colname], label='train_' + colname)
        ax.plot(results['epoch'], results['val_' + colname], label='val_' + colname)
        ax.set_xlabel('epoch')
        ax.set_ylabel(colname)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    # save graph as a file
    filename = op.join(opts.DATAPATH_CKP, ckpt_name, 'history.png')
    fig.savefig(filename, dpi=100)


def save_reconstruction_samples(model, dataset, ckpt_name, epoch):
    views = make_reconstructed_views(model, dataset)
    savepath = op.join(opts.DATAPATH_CKP, ckpt_name, 'reconimg')
    if not op.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)
    for i, view in enumerate(views):
        filename = op.join(savepath, f"ep{epoch:03d}_{i:02d}.png")
        cv2.imwrite(filename, view)


def make_reconstructed_views(model, dataset):
    recon_views = []
    for i, features in enumerate(dataset):
        predictions = model(features['image'])
        pred_disp_ms = predictions['disp_ms']
        pred_pose = predictions['pose']
        pred_depth_ms = uf.disp_to_depth_tensor(pred_disp_ms)
        print("predicted snippet poses:\n", pred_pose[0].numpy())

        # reconstruct target image
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        true_target_ms = uf.multi_scale_like(target_image, pred_disp_ms)
        synth_target_ms = SynthesizeMultiScale()(source_image, intrinsic, pred_depth_ms, pred_pose)

        # make stacked image of [true target, reconstructed target, source image, predicted depth] in 1/1 scale
        sclidx = 0
        view1 = uf.make_view(true_target_ms[sclidx], synth_target_ms[sclidx],
                             pred_depth_ms[sclidx], source_image,
                             batidx=0, srcidx=0, verbose=False)
        recon_views.append(view1)
        if i >= 10:
            break

    return recon_views


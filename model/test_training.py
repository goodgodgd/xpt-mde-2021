import os.path as op
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

import settings
from config import opts
import utils.util_funcs as uf
from tfrecords.tfrecord_reader import TfrecordGenerator
from model.model_builder import create_model
from model.model_main import train, set_configs, try_load_weights, read_previous_epoch
from model.loss_and_metric import disp_to_depth, multi_scale_like
from model.synthesize_batch import synthesize_batch_multi_scale


def test_train():
    model_name = "vode1"
    check_epochs = [1] + [10]*5
    dst_epoch = 0
    for i, epochs in enumerate(check_epochs):
        dst_epoch += epochs
        # train a few epochs
        train(train_dir_name="kitti_raw_test", val_dir_name="kitti_raw_test",
              model_name=model_name, learning_rate=0.0002, final_epoch=dst_epoch)

        # assert metric is improved?
        assert_metric_improved(model_name, dst_epoch - epochs, dst_epoch)

        # show reconstructed image
        predict_and_show(model_name, test_dir_name="kitti_raw_test")


def assert_metric_improved(model_name, last_epoch, curr_epoch):
    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.txt')
    history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
    print("history\n", history)
    history = history.values.astype(np.float)
    last_error = history[last_epoch - 1] if last_epoch > 0 else np.ones(7) * 1000
    curr_error = history[curr_epoch - 1]
    assert (curr_error < last_error).all(), f"curr error: {curr_error} \nlast error: {last_error}"
    print("!!! test_train passed")


def predict_and_show(model_name, test_dir_name):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()
    # keep the last model (function static variable)
    predict_and_show.last_imgs = getattr(predict_and_show, 'last_imgs', [])
    last_imgs = predict_and_show.last_imgs
    curr_imgs = []

    for i, features in enumerate(dataset):
        predictions = model(features['image'])
        pred_disp_ms = predictions['disp_ms']
        pred_pose = predictions['pose']
        pred_depth_ms = disp_to_depth(pred_disp_ms)

        disp = pred_disp_ms[0][0].numpy()
        print("disp\n", disp[80:85, 200:210, 0])

        # reconstruct target image
        stacked_image = features['image']
        intrinsic = features['intrinsic']
        source_image, target_image = uf.split_into_source_and_target(stacked_image)
        true_target_ms = multi_scale_like(target_image, pred_disp_ms)

        synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, pred_depth_ms, pred_pose)

        # make stacked image of [true target, recon target, recon target in prev training, source image]
        #   in 1/1 scale
        lastimg = last_imgs[i][0] if last_imgs else None
        view1, curimg1 = make_view(true_target_ms, synth_target_ms, pred_depth_ms, source_image, lastimg, sclidx=0, batidx=0, srcidx=0)
        #   in 1/4 scale
        lastimg = last_imgs[i][1] if last_imgs else None
        view2, curimg2 = make_view(true_target_ms, synth_target_ms, pred_depth_ms, source_image, lastimg, sclidx=2, batidx=0, srcidx=0)
        cv2.imshow("synth scale 1", view1)
        cv2.imshow("synth scale 1/4", view2)
        cv2.waitKey()
        curr_imgs.append([curimg1, curimg2])

        if i > 3:
            break

    predict_and_show.last_imgs = curr_imgs


def make_view(true_target_ms, synth_target_ms, pred_depth_ms, source_image, last_image, sclidx, batidx, srcidx):
    dsize = (opts.IM_HEIGHT, opts.IM_WIDTH)
    location = (20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1

    predim = tf.image.resize(synth_target_ms[sclidx][batidx, srcidx], size=dsize, method="nearest")
    predim = uf.to_uint8_image(predim).numpy()
    cv2.putText(predim, 'reconstructed target image', location, font, font_scale, color, thickness)

    trueim = tf.image.resize(true_target_ms[sclidx][batidx], size=dsize, method="nearest")
    trueim = uf.to_uint8_image(trueim).numpy()
    cv2.putText(trueim, 'true target image', location, font, font_scale, color, thickness)

    dpthim = tf.image.resize(pred_depth_ms[sclidx][batidx], size=dsize, method="nearest")
    depth = dpthim.numpy()
    print("depths\n", depth[80:85, 200:210, 0])
    dpthim = tf.clip_by_value(dpthim, 0., 10.) / 10.
    dpthim = tf.image.convert_image_dtype(dpthim, dtype=tf.uint8).numpy()
    dpthim = cv2.cvtColor(dpthim, cv2.COLOR_GRAY2BGR)

    sourim = uf.to_uint8_image(source_image).numpy()
    sourim = sourim[batidx, opts.IM_HEIGHT * srcidx:opts.IM_HEIGHT * (srcidx + 1)]
    cv2.putText(sourim, 'source image', location, font, font_scale, color, thickness)

    lastim = last_image if last_image else predim
    view = np.concatenate([trueim, dpthim, predim, lastim, sourim], axis=0)
    return view, predim


def test_train_disparity():
    """
    DEPRECATED: this function can be executed only when getting model_builder.py back to the below commit
    commit: 68612cb3600cfc934d8f26396b51aba0622ba357
    """
    model_name = "vode1"
    check_epochs = [1] + [5]*5
    dst_epoch = 0
    for i, epochs in enumerate(check_epochs):
        dst_epoch += epochs
        # train a few epochs
        train(train_dir_name="kitti_raw_test", val_dir_name="kitti_raw_test",
              model_name=model_name, learning_rate=0.0002, final_epoch=dst_epoch)

        check_disparity(model_name, "kitti_raw_test")


def check_disparity(model_name, test_dir_name):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()

    for i, features in enumerate(dataset):
        predictions = model.predict(features['image'])

        pred_disp_s1 = predictions[0]
        pred_disp_s4 = predictions[2]
        pred_pose = predictions[4]

        for k, conv in enumerate(predictions[5:]):
            print(f"conv{k} stats", np.mean(conv), np.std(conv), np.quantile(conv, [0, 0.2, 0.5, 0.8, 1.0]))

        batch, height, width, _ = pred_disp_s1.shape
        view_y, view_x = int(height * 0.3), int(width * 0.3)
        print("view pixel", view_y, view_x)
        print(f"disp scale 1, {pred_disp_s1.shape}\n", pred_disp_s1[0, view_y:view_y+50:10, view_x:view_x+100:10, 0])
        batch, height, width, _ = pred_disp_s4.shape
        view_y, view_x = int(height * 0.5), int(width * 0.3)
        print(f"disp scale 1/4, {pred_disp_s4.shape}\n", pred_disp_s4[0, view_y:view_y+50:10, view_x:view_x+100:10, 0])
        print("pose\n", pred_pose[0, 0])
        if i > 5:
            break


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    test_train_disparity()
    # test_train()

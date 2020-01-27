import os
import os.path as op
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2

import settings
from config import opts
from model.build_model.model_base import ModelBuilderBase
from model.synthesize_batch import synthesize_batch_multi_scale
from tfrecords.tfrecord_reader import TfrecordGenerator
import utils.util_funcs as uf
from utils.util_class import TrainException
import model.loss_and_metric as lm


# TODO: 클래스로 바꾸고 model_name, pose_valid 같은 멤버 변수 공통 사용

def train_by_user_interaction():
    options = {"train_dir_name": "kitti_raw_train",
               "val_dir_name": "kitti_raw_test",
               "model_name": "vode_model",
               "learning_rate": 0.0002,
               "final_epoch": opts.EPOCHS}

    print("\n===== Select training options")

    print(f"Default options:")
    for key, value in options.items():
        print(f"\t{key} = {value}")
    print("\nIf you are happy with default options, please press enter")
    print("Otherwise, please press any other key")
    select = input()

    if select == "":
        print(f"You selected default options.")
    else:
        message = "Type 1 or 2 to specify dataset: 1) kitti_raw, 2) kitti_odom"
        ds_id = uf.input_integer(message, 1, 2)
        if ds_id == 1:
            options["train_dir_name"] = "kitti_raw_train"
            options["val_dir_name"] = "kitti_raw_test"
        elif ds_id == 2:
            options["train_dir_name"] = "kitti_odom_train"
            options["val_dir_name"] = "kitti_odom_test"
        else:
            print("invalid option, dataset selection stay as default")

        print("Type model_name: dir name under opts.DATAPATH_CKP to save or load model")
        options["model_name"] = input()

        message = "Type learning_rate: learning rate"
        options["learning_rate"] = uf.input_float(message, 0, 10000)
        message = "Type final_epoch: number of epochs to train model upto"
        options["final_epoch"] = uf.input_integer(message, 0, 10000)

    print("Training options:", options)
    train(**options)


def train(train_dir_name, val_dir_name, model_name, learning_rate, final_epoch):
    initial_epoch = uf.read_previous_epoch(model_name)
    if final_epoch <= initial_epoch:
        raise TrainException("!! final_epoch <= initial_epoch, no need to train")

    set_configs(model_name)
    model_builder = ModelBuilderBase(need_resize=False)
    model = model_builder.create_model()
    model = try_load_weights(model, model_name)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dir_name), shuffle=True).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dir_name), shuffle=False).get_generator()
    steps_per_epoch = uf.count_steps(train_dir_name)

    print(f"\n\n========== START TRAINING ON {model_name} ==========")
    for epoch in range(initial_epoch, final_epoch):
        print(f"========== Start epoch: {epoch}/{final_epoch} ==========")
        result_train = train_an_epoch_graph(model, dataset_train, optimizer, steps_per_epoch)
        print(f"\n[Train Epoch MEAN], loss={result_train[0]:1.4f}, "
              f"metric={result_train[1]:1.4f}, {result_train[2]:1.4f}")

        result_val = validate_an_epoch_graph(model, dataset_val, steps_per_epoch)
        print(f"\n[Val Epoch MEAN],   loss={result_val[0]:1.4f}, "
              f"metric={result_val[1]:1.4f}, {result_val[2]:1.4f}")

        save_model(model, model_name, result_val[1])
        save_log(epoch, result_train, result_val, model_name)
        if epoch % 10 == 0:
            save_reconstruction_samples(model, dataset_val, model_name, epoch)


def set_configs(model_name):
    np.set_printoptions(precision=3, suppress=True)
    if not op.isdir(op.join(opts.DATAPATH_CKP, model_name)):
        os.makedirs(op.join(opts.DATAPATH_CKP, model_name), exist_ok=True)

    # set gpu configs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def try_load_weights(model, model_name, weight_name='latest.h5'):
    if model_name:
        model_file_path = op.join(opts.DATAPATH_CKP, model_name, weight_name)
        if op.isfile(model_file_path):
            print("===== load model weights", model_file_path)
            model.load_weights(model_file_path)
        else:
            print("===== train from scratch", model_file_path)
    return model


# Eager training is slow ...
def train_an_epoch_eager(model, dataset, optimizer, steps_per_epoch):
    results = []
    # tf.data.Dataset object is reusable after a full iteration, check test_reuse_dataset()
    for step, features in enumerate(dataset):
        start = time.time()
        with tf.GradientTape() as tape:
            preds = model(features['image'])
            loss = lm.compute_loss_vode(preds, features)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_num = loss.numpy().mean()
        trjerr, roterr = get_metric_pose(preds, features)
        results.append((loss_num, trjerr, roterr))
        uf.print_progress_status(f"\tTraining (eager) {step}/{steps_per_epoch} steps, loss={loss_num:1.4f}, "
                                 f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    mean_res = np.array(results).mean(axis=0)
    return mean_res


# Graph training is faster than eager training TWO TIMES!!
def train_an_epoch_graph(model, dataset, optimizer, steps_per_epoch):
    results = []
    # tf.data.Dataset object is reusable after a full iteration, check test_reuse_dataset()
    for step, features in enumerate(dataset):
        start = time.time()
        preds, loss = train_a_batch(model, features, optimizer)

        trjerr, roterr = get_metric_pose(preds, features)
        results.append((loss.numpy(), trjerr, roterr))
        uf.print_progress_status(f"\tTraining (graph) {step}/{steps_per_epoch} steps, loss={loss.numpy():1.4f}, "
                                 f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    results = np.stack(results, axis=0)
    mean_res = results.mean(axis=0)
    return mean_res


@tf.function
def train_a_batch(model, features, optimizer):
    with tf.GradientTape() as tape:
        # NOTE! preds = {"disp_ms": ..., "pose": ...} = model(image)
        preds = model(features['image'])
        loss = lm.compute_loss_vode(preds, features)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    loss_mean = tf.reduce_mean(loss)
    return preds, loss_mean


def validate_an_epoch_eager(model, dataset, steps_per_epoch):
    results = []
    # tf.data.Dataset 객체는 한번 쓴 후에도 다시 iteration 가능, test_reuse_dataset() 참조
    for step, features in enumerate(dataset):
        start = time.time()
        preds = model(features['image'])
        loss = lm.compute_loss_vode(preds, features)

        loss_num = loss.numpy().mean()
        trjerr, roterr = get_metric_pose(preds, features)
        results.append((loss_num, trjerr, roterr))
        uf.print_progress_status(f"\tValidating (eager) {step}/{steps_per_epoch} steps, loss={loss_num:1.4f}, "
                                 f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    mean_res = np.array(results).mean(axis=0)
    return mean_res


def validate_an_epoch_graph(model, dataset, steps_per_epoch):
    results = []
    for step, features in enumerate(dataset):
        start = time.time()
        preds, loss = validate_a_batch(model, features)

        trjerr, roterr = get_metric_pose(preds, features)
        results.append((loss.numpy(), trjerr, roterr))
        uf.print_progress_status(f"\tValidating (graph) {step}/{steps_per_epoch} steps, loss={loss.numpy():1.4f}, "
                                 f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    results = np.stack(results, axis=0)
    mean_res = results.mean(axis=0)
    return mean_res


@tf.function
def validate_a_batch(model, features):
    preds = model(features['image'])
    loss = lm.compute_loss_vode(preds, features)
    loss_mean = tf.reduce_mean(loss)
    return preds, loss_mean


def get_metric_pose(preds, features):
    if "pose_gt" in features:
        trjerr, roterr = lm.compute_metric_pose(preds['pose'], features['pose_gt'])
        return trjerr.numpy(), roterr.numpy()
    else:
        return 0, 0


def save_log(epoch, results_train, results_val, model_name):
    """
    :param epoch: current epoch
    :param results_train: (loss, metric_trj, metric_rot) from train data
    :param results_val: (loss, metric_trj, metric_rot) from validation data
    :param model_name: model directory name
    """
    results = np.concatenate([[epoch], results_train, results_val], axis=0)
    results = np.expand_dims(results, 0)
    columns = ['epoch', 'train_loss', 'train_metric_trj', 'train_metric_rot', 'val_loss', 'val_metric_trj', 'val_metric_rot']
    results = pd.DataFrame(data=results, columns=columns)
    results['epoch'] = results['epoch'].astype(int)

    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.txt')
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
    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.png')
    fig.savefig(filename, dpi=100)


def save_model(model, model_name, val_loss):
    """
    :param model: nn model object
    :param model_name: model directory name
    :param val_loss: current validation loss
    """
    # save the latest model
    save_model_weights(model, model_name, 'latest.h5')
    # save the best model (function static variable)
    save_model.best = getattr(save_model, 'best', 10000)
    if val_loss < save_model.best:
        save_model_weights(model, model_name, 'best.h5')
        save_model.best = val_loss


def save_model_weights(model, model_name, weights_name):
    model_dir_path = op.join(opts.DATAPATH_CKP, model_name)
    if not op.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)
    model_file_path = op.join(opts.DATAPATH_CKP, model_name, weights_name)
    model.save_weights(model_file_path)


def predict_by_user_interaction():
    options = {"test_dir_name": "kitti_raw_test",
               "model_name": "vode_model",
               "weights_name": "latest.h5"
               }

    print("\n===== Select prediction options")

    print(f"Default options:")
    for key, value in options.items():
        print(f"\t{key} = {value}")
    print("\nIf you are happy with default options, please press enter")
    print("Otherwise, please press any other key")
    select = input()

    if select == "":
        print(f"You selected default options.")
    else:
        message = "Type 1 or 2 to specify dataset: 1) kitti_raw, 2) kitti_odom"
        ds_id = uf.input_integer(message, 1, 2)
        if ds_id == 1:
            options["test_dir_name"] = "kitti_raw_test"
        elif ds_id == 2:
            options["test_dir_name"] = "kitti_odom_test"
        else:
            raise ValueError("Wrong value for dataset")

        print("Type model_name: dir name under opts.DATAPATH_CKP")
        options["model_name"] = input()
        print("Type weights_name: load weights from [model_name/weights_name]")
        options["weights_name"] = input()

    print("Prediction options:", options)
    predict(**options)


def save_reconstruction_samples(model, dataset, model_name, epoch):
    views = make_reconstructed_views(model, dataset)
    savepath = op.join(opts.DATAPATH_CKP, model_name, 'reconimg')
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
        synth_target_ms = synthesize_batch_multi_scale(source_image, intrinsic, pred_depth_ms, pred_pose)

        # make stacked image of [true target, reconstructed target, source image, predicted depth] in 1/1 scale
        sclidx = 0
        view1 = uf.make_view(true_target_ms[sclidx], synth_target_ms[sclidx], pred_depth_ms[sclidx],
                             source_image, batidx=0, srcidx=0)
        recon_views.append(view1)
        if i >= 10:
            break

    return recon_views


def predict(test_dir_name, model_name, weight_name):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name, weight_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()
    # [disp_s1, disp_s2, disp_s4, disp_s8, pose] = model.predict({"image": ...})
    predictions = model.predict(dataset)
    for pred in predictions:
        print(f"prediction shape={pred.shape}")

    pred_disp = predictions[0]
    pred_pose = predictions[4]
    save_predictions(model_name, pred_disp, pred_pose)


def save_predictions(model_name, pred_disp, pred_pose):
    pred_dir_path = op.join(opts.DATAPATH_PRD, model_name)
    os.makedirs(pred_dir_path, exist_ok=True)
    print(f"save depth in {pred_dir_path}, shape={pred_disp[0].shape}")
    np.save(op.join(pred_dir_path, "depth.npy"), pred_disp)
    print(f"save pose in {pred_dir_path}, shape={pred_pose.shape}")
    np.save(op.join(pred_dir_path, "pose.npy"), pred_pose)


# ==================== tests ====================

def run_train_default():
    train(train_dir_name="kitti_raw_test", val_dir_name="kitti_raw_test",
          model_name="vode2", learning_rate=0.0002, final_epoch=40)


def run_pred_default():
    predict(test_dir_name="kitti_raw_test", model_name="vode1", weight_name="best.h5")


def test_model_output():
    """
    check model output formats according to prediction methods
    1) preds = model(image_tensor) -> dict('disp_ms': disp_ms, 'pose': pose)
        disp_ms: list of [batch, height/scale, width/scale, 1]
        pose: [batch, num_src, 6]
    2) preds = model.predict(image_tensor) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    3) preds = model.predict({'image':, ...}) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    :return:
    """
    model_name = "vode1"
    test_dir_name = "kitti_raw_test"
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")
    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()

    print("===== check model output shape from 'preds = model(image)'")
    for i, features in enumerate(dataset):
        preds = model(features['image'])
        print("output dict keys:", list(preds.keys()))
        disp_ms = preds['disp_ms']
        pose = preds['pose']
        for disp in disp_ms:
            print("disparity shape:", disp.get_shape().as_list())
        print("pose shape:", pose.get_shape().as_list())
        break

    print("===== it does NOT work: 'preds = model({'image': ...})'")
    # for i, features in enumerate(dataset):
    #     preds = model(features)

    print("===== check model output shape from 'preds = model.predict(image)'")
    for i, features in enumerate(dataset):
        preds = model.predict(features['image'])
        for pred in preds:
            print("predict output shape:", pred.shape)
        break

    print("===== check model output shape from 'preds = model.predict({'image': ...})'")
    for i, features in enumerate(dataset):
        preds = model.predict(features)
        for pred in preds:
            print("predict output shape:", pred.shape)
        break


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
    """
    DEPRECATED: this function can be executed only when getting model_builder.py back to the below commit
    commit: 68612cb3600cfc934d8f26396b51aba0622ba357
    """
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
    # test_count_steps()
    run_train_default()
    run_pred_default()
    # test_model_output()

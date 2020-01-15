import os
import os.path as op
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import time

import settings
from config import opts
from model.model_builder import create_model
from tfrecords.tfrecord_reader import TfrecordGenerator
from utils.util_funcs import input_integer, input_float, print_progress_status
import model.loss_and_metric as lm


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
        ds_id = input_integer(message, 1, 2)
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
        options["learning_rate"] = input_float(message, 0, 10000)
        message = "Type final_epoch: number of epochs to train model upto"
        options["final_epoch"] = input_integer(message, 0, 10000)

    print("Training options:", options)
    train(**options)


def train(train_dir_name, val_dir_name, model_name, learning_rate, final_epoch):
    initial_epoch = read_previous_epoch(model_name)
    if final_epoch <= initial_epoch:
        print("!! final_epoch <= initial_epoch, no need to train")
        return

    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dir_name), shuffle=True).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dir_name), shuffle=False).get_generator()
    steps_per_epoch = count_steps(train_dir_name)

    print(f"\n\n========== START TRAINING ON {model_name} ==========")
    for epoch in range(initial_epoch, final_epoch):
        print(f"========== Start epoch: {epoch}/{final_epoch} ==========")
        result_train = train_an_epoch_graph(model, dataset_train, optimizer, steps_per_epoch)
        print(f"\n[Train Epoch MEAN], loss={result_train[0]:1.4f}, "
              f"metric={result_train[1]:1.4f}, {result_train[2]:1.4f}")

        result_val = validate_an_epoch_graph(model, dataset_val, steps_per_epoch)
        print(f"\n[Val Epoch MEAN],   loss={result_val[0]:1.4f}, "
              f"metric={result_val[1]:1.4f}, {result_val[2]:1.4f}")

        save_log(epoch, result_train, result_val, model_name)
        save_model(model, model_name, result_val[1])


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


def try_load_weights(model, model_name):
    if model_name:
        model_file_path = op.join(opts.DATAPATH_CKP, model_name, "latest.h5")
        if op.isfile(model_file_path):
            print("===== load model weights", model_file_path)
            model.load_weights(model_file_path)
        else:
            print("===== train from scratch", model_file_path)
    return model


def count_steps(dataset_dir):
    tfrpath = op.join(opts.DATAPATH_TFR, dataset_dir)
    with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
        config = json.load(fr)
    frames = config['length']
    steps = frames // opts.BATCH_SIZE
    print(f"[count steps] frames={frames}, steps={steps}")
    return steps


def read_previous_epoch(model_name):
    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.txt')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            return 0
        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        return 0


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

        trjerr, roterr = lm.compute_metric_pose(preds['pose'], features['pose_gt'])
        loss_num = loss.numpy().mean()
        results.append((loss_num, trjerr.numpy(), roterr.numpy()))
        print_progress_status(f"\tTraining (eager) {step}/{steps_per_epoch} steps, loss={loss_num:1.4f}, "
                              f"metric={trjerr.numpy():1.4f}, {roterr.numpy():1.4f}, time={time.time() - start:1.4f} ...")

    mean_res = np.array(results).mean(axis=0)
    return mean_res


# Graph training is faster than eager training TWO TIMES!!
def train_an_epoch_graph(model, dataset, optimizer, steps_per_epoch):
    results = []
    # tf.data.Dataset object is reusable after a full iteration, check test_reuse_dataset()
    for step, features in enumerate(dataset):
        start = time.time()
        result = train_a_batch(model, features, optimizer)
        result = result.numpy()
        results.append(result)
        print_progress_status(f"\tTraining (graph) {step}/{steps_per_epoch} steps, loss={result[0]:1.4f}, "
                              f"metric={result[1]:1.4f}, {result[2]:1.4f}, time={time.time() - start:1.4f} ...")

    results = np.stack(results, axis=0)
    mean_res = results.mean(axis=0)
    return mean_res


@tf.function
def train_a_batch(model, features, optimizer):
    with tf.GradientTape() as tape:
        # NOTE! preds = {"disp_ms": ..., "pose": ...} = model(image)
        #       preds = [disp_s1, disp_s2, disp_s4, disp_s8, pose] = model.predict({"image": ...})
        preds = model(features['image'])
        loss = lm.compute_loss_vode(preds, features)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    trjerr, roterr = lm.compute_metric_pose(preds['pose'], features['pose_gt'])
    loss_mean = tf.reduce_mean(loss)
    return tf.stack([loss_mean, trjerr, roterr], 0)


def validate_an_epoch_eager(model, dataset, steps_per_epoch):
    results = []
    # tf.data.Dataset 객체는 한번 쓴 후에도 다시 iteration 가능, test_reuse_dataset() 참조
    for step, features in enumerate(dataset):
        start = time.time()
        preds = model(features['image'])
        loss = lm.compute_loss_vode(preds, features)
        trjerr, roterr = lm.compute_metric_pose(preds['pose'], features['pose_gt'])
        loss_num = loss.numpy().mean()
        results.append((loss_num, trjerr, roterr))
        print_progress_status(f"\tValidating {step}/{steps_per_epoch} steps, loss={loss_num:1.4f}, "
                              f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    mean_res = np.array(results).mean(axis=0)
    return mean_res


def validate_an_epoch_graph(model, dataset, steps_per_epoch):
    results = []
    for step, features in enumerate(dataset):
        start = time.time()
        result = validate_a_batch(model, features)
        result = result.numpy()
        results.append(result)
        print_progress_status(f"\tValidating (graph) {step}/{steps_per_epoch} steps, loss={result[0]:1.4f}, "
                              f"metric={result[1]:1.4f}, {result[2]:1.4f}, time={time.time() - start:1.4f} ...")

    results = np.stack(results, axis=0)
    mean_res = results.mean(axis=0)
    return mean_res


@tf.function
def validate_a_batch(model, features):
    preds = model(features['image'])
    loss = lm.compute_loss_vode(preds, features)
    trjerr, roterr = lm.compute_metric_pose(preds['pose'], features['pose_gt'])
    loss_mean = tf.reduce_mean(loss)
    return tf.stack([loss_mean, trjerr, roterr], 0)


def save_log(epoch, results_train, results_val, model_name):
    """
    :param epoch: current epoch
    :param results_train: list of (epoch, loss, metric) from train data
    :param results_val: list of (epoch, loss, metric) from validation data
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
    results.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')


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
        ds_id = input_integer(message, 1, 2)
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


def predict(test_dir_name, model_name, weights_name):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name, weights_name)
    model.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()
    # NOTE! preds = {"disp_ms": ..., "pose": ...} = model(image)
    #       preds = [disp_s1, disp_s2, disp_s4, disp_s8, pose] = model.predict({"image": ...})
    predictions = model.predict(dataset)
    for pred in predictions:
        print(f"prediction shape={pred.shape}")

    pred_disps_ms = predictions[0]
    pred_pose = predictions[1]
    print("predicted dispartiy shape:", pred_disps_ms[0].get_shape().as_list())
    print("predicted dispartiy shape:", pred_disps_ms[2].get_shape().as_list())
    print("predicted pose shape:", pred_pose.get_shape().as_list())
    # save_predictions(model_name, pred_disps_ms, pred_pose)


def save_predictions(model_name, pred_disps_ms, pred_pose):
    pred_dir_path = op.join(opts.DATAPATH_PRD, model_name)
    os.makedirs(pred_dir_path, exist_ok=True)
    print(f"save depth in {pred_dir_path}, shape={pred_disps_ms[0].shape}")
    np.save(op.join(pred_dir_path, "depth.npy"), pred_disps_ms)
    print(f"save pose in {pred_dir_path}, shape={pred_pose.shape}")
    np.save(op.join(pred_dir_path, "pose.npy"), pred_pose)


# ==================== tests ====================

def test_count_steps():
    steps = count_steps('kitti_raw_train')


def run_train_default():
    train(train_dir_name="kitti_raw_test", val_dir_name="kitti_raw_test",
          model_name="vode1", learning_rate=0.0002, final_epoch=10)


def run_pred_default():
    predict(test_dir_name="kitti_raw_test", model_name="vode1", weights_name="latest.h5")


def test_model_output():
    """
    check model output formats according to prediction methods
    1) preds = model(image_tensor) -> dict('disp_ms': disp_ms, 'pose': pose)
        disp_ms: list of [batch, height/scale, width/scale, 1]
        pose: [batch, num_src, 6]
    2) preds = model(image_tensor) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
    2) preds = model({'image':, ...}) -> [disp_s1, disp_s2, disp_s4, disp_s8, pose]
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
        disp_ms = preds['disp_ms']
        pose = preds['pose']
        for disp in disp_ms:
            print("disparity shape:", disp.get_shape().as_list())
        print("pose shape:", pose.get_shape().as_list())
        break

    print("===== check model output shape from 'preds = model.predict(image)'")
    for i, features in enumerate(dataset):
        preds = model.predict(features['image'])
        for pred in preds:
            print("predict output shape:", pred.shape)
        break

    print("===== check model output shape from 'preds = model.predict({'image': ...)'")
    for i, features in enumerate(dataset):
        preds = model.predict(features)
        for pred in preds:
            print("predict output shape:", pred.shape)
        break


if __name__ == "__main__":
    test_model_output()
    # test_count_steps()
    # run_train_default()
    # run_pred_default()

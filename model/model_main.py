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
               "src_weights_name": "latest.h5",
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
        print("Type src_weights_name: load weights from [model_name/src_weights_name]")
        options["src_weights_name"] = input()

        message = "Type learning_rate: learning rate"
        options["learning_rate"] = input_float(message, 0, 10000)
        message = "Type final_epoch: number of epochs to train model upto"
        options["final_epoch"] = input_integer(message, 0, 10000)

    print("Training options:", options)
    train(**options)


def train(train_dir_name, val_dir_name, model_name, src_weights_name, learning_rate, final_epoch):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name, src_weights_name)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dir_name), shuffle=True).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dir_name), shuffle=False).get_generator()
    steps_per_epoch = count_steps(train_dir_name)
    initial_epoch = read_previous_epoch(model_name)
    results_train, results_val = [], []

    print(f"\n\n========== START TRAINING ON {model_name} ==========")
    for epoch in range(initial_epoch, final_epoch):
        print(f"========== Start epoch: {epoch}/{final_epoch} ==========")
        result = train_an_epoch_graph(model, dataset_train, optimizer, steps_per_epoch)
        print(f"\n[Train Epoch MEAN], loss={result[0]:1.4f}, metric={result[1]:1.4f} {result[2]:1.4f}")
        results_train.append(np.insert(result, 0, epoch))

        result = validate_an_epoch_graph(model, dataset_val, steps_per_epoch)
        print(f"\n[Val Epoch MEAN],   loss={result[0]:1.4f}, metric={result[1]:1.4f} {result[2]:1.4f}")
        results_val.append(np.insert(result, 0, epoch))

        save_log(results_train, results_val, model_name)
        save_model(model, model_name, results_val[-1][1])


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


def try_load_weights(model, model_name, weights_name):
    if model_name and weights_name:
        model_file_path = op.join(opts.DATAPATH_CKP, model_name, weights_name)
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
        print(f"start from epoch {prev_epoch + 1}")
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


def save_log(results_train, results_val, model_name):
    """
    :param results_train: list of (epoch, loss, metric) from train data
    :param results_val: list of (epoch, loss, metric) from validation data
    :param model_name: model directory name
    """
    results_train = np.array(results_train)
    results_val = np.array(results_val)
    results = np.concatenate([results_train, results_val[:, 1:]], axis=1)
    columns = ['epoch', 'train_loss', 'train_metric_trj', 'train_metric_rot', 'val_loss', 'val_metric_trj', 'val_metric_rot']
    results = pd.DataFrame(data=results, columns=columns)
    results['epoch'] = results['epoch'].astype(int)

    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.txt')
    # if the file existed, append new data to it
    if op.isfile(filename):
        existing = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        results = pd.concat([results, existing], axis=0)
        results = results.drop_duplicates(subset='epoch', keep='first')
    results.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')


def save_model(model, model_name, val_loss):
    """
    :param model: nn model object
    :param model_name: model directory name
    :param val_loss: current validation loss
    """
    # save the latest model
    save_model_weights(model, model_name, 'latest.h5')
    # save the best model
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
    predictions = model.predict(dataset)
    for pred in predictions:
        print(f"prediction shape={pred.shape}")

    pred_depth = predictions[0]
    pred_pose = predictions[-1]
    save_predictions(model_name, pred_depth, pred_pose)


def save_predictions(model_name, pred_depth, pred_pose):
    pred_dir_path = op.join(opts.DATAPATH_PRD, model_name)
    os.makedirs(pred_dir_path, exist_ok=True)
    print(f"save depth in {pred_dir_path}, shape={pred_depth.shape}")
    np.save(op.join(pred_dir_path, "depth.npy"), pred_depth)
    print(f"save pose in {pred_dir_path}, shape={pred_pose.shape}")
    np.save(op.join(pred_dir_path, "pose.npy"), pred_pose)


# ==================== tests ====================

def test_count_steps():
    steps = count_steps('kitti_raw_train')


def run_train_default():
    train(train_dir_name="kitti_raw_test", val_dir_name="kitti_raw_test",
          model_name="vode1", src_weights_name="latest.h5",
          learning_rate=0.0002, final_epoch=10)


def run_pred_default():
    predict(test_dir_name="kitti_raw_test", model_name="vode1", weights_name="latest.h5")


if __name__ == "__main__":
    # test_count_steps()
    run_train_default()
    # run_pred_default()

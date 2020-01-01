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
               "initial_epoch": 0,
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
        print("Type src_weights_name: load weights from {model_name/src_weights_name}")
        options["src_weights_name"] = input()

        message = "Type learning_rate: learning rate"
        options["learning_rate"] = input_float(message, 0, 10000)
        message = "Type initial_epoch: number of epochs previously trained"
        options["initial_epoch"] = input_integer(message, 0, 10000)
        message = "Type final_epoch: number of epochs to train model upto"
        options["final_epoch"] = input_integer(message, 0, 10000)

    print("Training options:", options)
    train(**options)


def train(train_dir_name, val_dir_name, model_name, src_weights_name, learning_rate,
          initial_epoch, final_epoch):
    set_configs(model_name)
    model = create_model()
    model = try_load_weights(model, model_name, src_weights_name)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dir_name), shuffle=True).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dir_name), shuffle=False).get_generator()
    steps_per_epoch = count_steps(train_dir_name)
    results_train, results_val = [], []

    print(f"\n\n========== START TRAINING ON {model_name} ==========")
    for epoch in range(initial_epoch, final_epoch):
        print(f"========== Start epoch: {epoch}/{final_epoch} ==========")
        results = train_an_epoch(epoch, model, dataset_train, optimizer, steps_per_epoch)
        results_train.append(results)

        results = validate_an_epoch(epoch, model, dataset_val, steps_per_epoch)
        results_val.append(results)

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


def train_an_epoch(epoch, model, dataset, optimizer, steps_per_epoch):
    results = []
    # tf.data.Dataset 객체는 한번 쓴 후에도 다시 iteration 가능, test_reuse_dataset() 참조
    for step, features in enumerate(dataset):
        start = time.time()
        with tf.GradientTape() as tape:
            preds = model(features['image'])
            loss = lm.compute_loss_vode(preds, features)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        trjerr, roterr = lm.compute_metric_pose(preds, features)
        loss_num = loss.numpy().mean()
        results.append((loss_num, trjerr, roterr))
        print_progress_status(f"\tTraining {step}/{steps_per_epoch} steps, loss={loss_num:1.4f}, "
                              f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    mean_res = np.array(results).mean(axis=0)
    print(f"\n[Epoch {epoch:03d}], train loss={mean_res[0]:1.4f}, metric={mean_res[1]:1.4f} {mean_res[2]:1.4f}")
    return epoch, mean_res[0], mean_res[1]


def validate_an_epoch(epoch, model, dataset, steps_per_epoch):
    results = []
    # tf.data.Dataset 객체는 한번 쓴 후에도 다시 iteration 가능, test_reuse_dataset() 참조
    for step, features in enumerate(dataset):
        start = time.time()
        preds = model(features['image'])
        loss = lm.compute_loss_vode(preds, features)
        trjerr, roterr = lm.compute_metric_pose(preds, features)
        loss_num = loss.numpy().mean()
        results.append((loss_num, trjerr, roterr))
        print_progress_status(f"\tEvaluating {step}/{steps_per_epoch} steps, loss={loss_num:1.4f}, "
                              f"metric={trjerr:1.4f}, {roterr:1.4f}, time={time.time() - start:1.4f} ...")

    mean_res = np.array(results).mean(axis=0)
    print(f"\n[Epoch {epoch:03d}], val loss={mean_res[0]:1.4f}, metric={mean_res[1]:1.4f} {mean_res[2]:1.4f}")
    return epoch, mean_res[0], mean_res[1]


def save_log(results_train, results_val, model_name):
    """
    :param results_train: list of (epoch, loss, metric) from train data
    :param results_val: list of (epoch, loss, metric) from validation data
    :param model_name: model directory name
    """
    results_train = np.array(results_train)
    results_val = np.array(results_val)
    results = np.concatenate([results_train, results_val[:, 1:]], axis=1)
    results = pd.DataFrame(data=results, columns=['epoch', 'train_loss', 'train_metric', 'val_loss', 'val_metric'])
    results['epoch'] = results['epoch'].astype(int)

    # save log
    filename = op.join(opts.DATAPATH_CKP, model_name, 'history.txt')
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
        if ds_id == 2:
            options["test_dir_name"] = "kitti_odom_test"

        print("Type model_name: dir name under opts.DATAPATH_CKP and opts.DATAPATH_PRD")
        options["model_name"] = input()
        print("Type weights_name: load weights from {model_name/src_weights_name}")
        options["weights_name"] = input()

    print("Prediction options:", options)
    predict(**options)


# TODO: pred model 만 나오는 create_model 로 수정
def predict(test_dir_name, model_name, weights_name):
    set_configs(model_name)

    model_pred, model_train = create_models()
    model_train = try_load_weights(model_train, model_name, weights_name)
    model_pred.compile(optimizer="sgd", loss="mean_absolute_error")

    dataset = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dir_name)).get_generator()
    predictions = model_pred.predict(dataset)
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
          learning_rate=0.0002, initial_epoch=0, final_epoch=10)


if __name__ == "__main__":
    # test_count_steps()
    run_train_default()

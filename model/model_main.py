import os
import os.path as op
import tensorflow as tf
import datetime
import numpy as np
from glob import glob
import pandas as pd

import settings
from config import opts
from model.model_builder import create_model
from tfrecords.tfrecord_reader import TfrecordGenerator
from utils.util_funcs import input_integer, input_float, print_progress
import model.loss_and_metric as lm


def train_by_user_interaction():
    options = {"train_dir_name": "kitti_raw_train",
               "val_dir_name": "kitti_raw_test",
               "model_name": "vode_model",
               "src_weights_name": "latest.h5",
               "dst_weights_name": "latest.h5",
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
        print("Type dst_weights_name: save weights to {model_name/dst_weights_name}")
        options["dst_weights_name"] = input()

        message = "Type learning_rate: learning rate"
        options["learning_rate"] = input_float(message, 0, 10000)
        message = "Type initial_epoch: number of epochs previously trained"
        options["initial_epoch"] = input_integer(message, 0, 10000)
        message = "Type final_epoch: number of epochs to train model upto"
        options["final_epoch"] = input_integer(message, 0, 10000)

    print("Training options:", options)
    train(**options)


def train_by_tape(train_dir_name, val_dir_name, model_name, src_weights_name, dst_weights_name,
                  learning_rate, initial_epoch, final_epoch):
    set_gpu_config()
    model = create_model()
    model = try_load_weights(model, model_name, src_weights_name)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dir_name), True, opts.EPOCHS).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dir_name), True, opts.EPOCHS).get_generator()
    steps_per_epoch = count_steps(train_dir_name)
    results_train = []
    results_val = []

    print(f"\n\n========== START TRAINING ON {model_name} ==========\n\n")
    for epoch in range(initial_epoch, final_epoch):
        print(f"\n========== Start epoch: {epoch} ==========")
        results = train_an_epoch(epoch, model, dataset_train, optimizer, steps_per_epoch)
        results_train.append(results)

        results = validate_an_epoch(epoch, model, dataset_val, steps_per_epoch)
        results_val.append(results)

        summarize_results(model, results_train, results_val)


def train_an_epoch(epoch, model, dataset, optimizer, steps_per_epoch):
    results = []

    # TODO: dataset은 한번 for loop 돌고나서 또 써도 되나?
    for step, features in enumerate(dataset):
        with tf.GradientTape() as tape:
            preds = model(features)
            # predictions = {"disp_ms": pred_disps_ms, "pose": pred_poses}
            loss = loss_vode(preds, features)
            # TODO: 스케일만 다른 y_true, y_pred 넣었을 때 값이 0 나오는지 테스트 함수 만들기
            pose_metric = metric_pose(preds, features)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        results.append((epoch, step, loss.numpy(), pose_metric.numpy()))

        if step % 200 == 0:
            mean_res = np.array(results[-200:-1]).mean(axis=0)
            print(f"Training {step}/{steps_per_epoch}, loss={mean_res[2]}, metric={mean_res[3]}")

        return results


def validate_an_epoch(epoch, model, dataset, steps_per_epoch):
    results = []

    # TODO: dataset은 한번 for loop 돌고나서 또 써도 되나?
    for step, features in enumerate(dataset):
        preds = model(features)
        # predictions = {"disp_ms": pred_disps_ms, "pose": pred_poses}
        loss = lm.loss_vode(preds, features)
        # TODO: 스케일만 다른 y_true, y_pred 넣었을 때 값이 0 나오는지 테스트 함수 만들기
        pose_metric = lm.metric_pose(preds, features)
        results.append((epoch, step, loss.numpy(), pose_metric.numpy()))

        if step % 200 == 0:
            mean_res = np.array(results[-200:-1]).mean(axis=0)
            print(f"Training {step}/{steps_per_epoch}, loss={mean_res[2]}, metric={mean_res[3]}")

        return results


def set_gpu_config():
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


def save_model_weights(model, model_name, weights_name):
    model_dir_path = op.join(opts.DATAPATH_CKP, model_name)
    if not op.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)
    model_file_path = op.join(opts.DATAPATH_CKP, model_name, weights_name)
    model.save_weights(model_file_path)


# TODO: tfrecord config 파일에 카운트 수 넣고 읽기
def count_steps(dataset_dir):
    srcpath = op.join(opts.DATAPATH_SRC, dataset_dir)
    files = glob(op.join(srcpath, "*/*.png"))
    frames = len(files)
    steps = frames // opts.BATCH_SIZE
    print(f"[count steps] frames={frames}, steps={steps}")
    return steps


def dump_history(history, model_name, initial_epoch):
    filename = op.join(opts.DATAPATH_CKP, model_name, "history.txt")
    hist_df = pd.DataFrame(history)
    if op.isfile(filename) and initial_epoch > 0:
        existing_df = pd.read_csv(filename)
        hist_df = pd.concat([existing_df, hist_df])
        print("concat hist")
    hist_df.to_csv(filename, encoding="utf-8", index=False, float_format="%1.3f")
    print("save history\n", hist_df)


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


def predict(test_dir_name, model_name, weights_name):
    set_gpu_config()

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


if __name__ == "__main__":
    train(train_dir_name="kitti_raw_train", val_dir_name="kitti_raw_test",
          model_name="vode1", src_weights_name="latest.h5", dst_weights_name="latest.h5",
          learning_rate=0.0002, initial_epoch=0, final_epoch=2)

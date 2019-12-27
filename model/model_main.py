import os
import os.path as op
import tensorflow as tf
import datetime
import numpy as np
from glob import glob
import pandas as pd

import settings
from config import opts
from model.model_builder import create_models
from tfrecords.tfrecord_reader import TfrecordGenerator
from utils.util_funcs import input_integer, input_float, print_progress


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


class LM:
    @staticmethod
    def loss_for_loss(y_true, y_pred):
        return y_pred

    @staticmethod
    def loss_for_metric(y_true, y_pred):
        return tf.constant(0, dtype=tf.float32)

    @staticmethod
    def metric_for_loss(y_true, y_pred):
        return tf.constant(0, dtype=tf.float32)

    @staticmethod
    def metric_for_metric(y_true, y_pred):
        return y_pred


def train(train_dir_name, val_dir_name, model_name, src_weights_name, dst_weights_name,
          learning_rate, initial_epoch, final_epoch):
    set_gpu_config()

    model_pred, model_train = create_models()
    model_train = try_load_weights(model_train, model_name, src_weights_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    losses = {"loss_out": LM.loss_for_loss, "metric_out": LM.loss_for_metric}
    metrics = {"loss_out": LM.metric_for_loss, "metric_out": LM.metric_for_metric}
    model_train.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dir_name), True, opts.EPOCHS).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dir_name), True, opts.EPOCHS).get_generator()
    callbacks = get_callbacks(model_name, dst_weights_name)
    steps_per_epoch = count_steps(train_dir_name)
    val_steps = np.clip(count_steps(train_dir_name)/2, 0, 100).astype(np.int32)

    print(f"\n\n========== START TRAINING ON {model_name} ==========\n\n")
    history = model_train.fit(dataset_train, epochs=final_epoch, callbacks=callbacks,
                              validation_data=dataset_val, steps_per_epoch=steps_per_epoch,
                              validation_steps=val_steps, initial_epoch=initial_epoch)

    if model_name:
        dump_history(history.history, model_name, initial_epoch)


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


def get_callbacks(model_name, weights_name):
    model_dir_path = op.join(opts.DATAPATH_CKP, model_name)
    best_ckpt_file = op.join(model_dir_path, "model-{epoch:02d}-{val_loss:.2f}.h5")
    regular_ckpt_file = op.join(model_dir_path, weights_name)
    os.makedirs(model_dir_path, exist_ok=True)
    log_dir = op.join(opts.DATAPATH_LOG, model_name)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt_file,
            monitor="val_loss",
            save_best_only=True,
            save_freq="epoch",
            save_weights_only=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=regular_ckpt_file,
            save_freq="epoch",
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        ),
    ]
    return callbacks


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

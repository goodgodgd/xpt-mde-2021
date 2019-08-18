import os
import os.path as op
import tensorflow as tf
import datetime
import numpy as np
from glob import glob
import json
import pandas as pd

import settings
from config import opts
from model.model_builder import create_models
from tfrecords.tfrecord_reader import TfrecordGenerator


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


def train(train_dirname, val_dirname, model_dir="", src_model_name="", dst_model_name="", initial_epoch=0):
    set_gpu_config()

    model_pred, model_train = create_models()
    model_train = try_load_weights(model_train, model_dir, src_model_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    losses = {"loss_out": LM.loss_for_loss, "metric_out": LM.loss_for_metric}
    metrics = {"loss_out": LM.metric_for_loss, "metric_out": LM.metric_for_metric}
    model_train.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    dataset_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dirname), shuffle=True, epochs=opts.EPOCHS).get_generator()
    dataset_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dirname), shuffle=True, epochs=opts.EPOCHS).get_generator()
    callbacks = get_callbacks(model_dir)
    steps_per_epoch = count_steps(train_dirname)
    val_steps = np.clip(count_steps(train_dirname)/2, 0, 100).astype(np.int32)

    print(f"\n\n\n========== START TRAINING ON {model_dir} ==========\n\n\n")
    history = model_train.fit(dataset_train, epochs=opts.EPOCHS, callbacks=callbacks,
                              validation_data=dataset_val, steps_per_epoch=steps_per_epoch,
                              validation_steps=val_steps, initial_epoch=initial_epoch)

    save_model_weights(model_train, model_dir, dst_model_name)
    if model_dir:
        dump_history(history.history, model_dir)


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


def try_load_weights(model, model_dir, model_name):
    if model_dir and model_name:
        model_file_path = op.join(opts.DATAPATH_CKP, model_dir, model_name)
        if op.isfile(model_file_path):
            print("===== load model", model_file_path)
            model.load_weights(model_file_path)
    return model


def save_model_weights(model, model_dir, model_name):
    model_dir_path = op.join(opts.DATAPATH_CKP, model_dir)
    if not op.isdir(model_dir_path):
        os.makedirs(model_dir_path, exist_ok=True)
    model_file_path = op.join(opts.DATAPATH_CKP, model_dir, model_name)
    model.save_weights(model_file_path)


def get_callbacks(model_dir):
    if model_dir:
        model_path = op.join(opts.DATAPATH_CKP, model_dir, "model-{epoch:02d}-{val_loss:.2f}.h5")
        log_dir = op.join(opts.DATAPATH_LOG, model_dir)
    else:
        nowtime = datetime.datetime.now()
        nowtime = nowtime.strftime("%m%d_%H%M%S")
        model_path = op.join(opts.DATAPATH_CKP, nowtime, "model-{epoch:02d}-{val_loss:.2f}.h5")
        log_dir = op.join(opts.DATAPATH_LOG, nowtime)

    if not op.isdir(model_path):
        os.makedirs(op.dirname(model_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
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


def dump_history(history, model_dir):
    df = pd.DataFrame(history)
    df.to_csv(op.join(opts.DATAPATH_CKP, model_dir, "history.txt"), float_format="%.3f")
    print("save history\n", df)


def predict():
    pass


def evaluate():
    pass


if __name__ == "__main__":
    for i in range(5):
        train("kitti_raw_train", "kitti_raw_test", model_dir=f"vode_model_{i}",
              src_model_name="weights.h5", dst_model_name="weights.h5", initial_epoch=0)

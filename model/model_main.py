import os
import os.path as op
import tensorflow as tf
import datetime
from glob import glob
import numpy as np

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


def train(train_dirname, val_dirname, src_model, dst_model):
    set_gpu_config()

    # model_train = None
    # if model_train is None:
    model_pred, model_train = create_models()

    if src_model:
        src_model_path = op.join(opts.DATAPATH_CKP, src_model)
        if op.isfile(src_model_path):
            print("===== load model", src_model_path)
            model_train.load_weights(src_model_path)
            # model_train = tf.keras.models.load_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    losses = {"loss_out": LM.loss_for_loss, "metric_out": LM.loss_for_metric}
    metrics = {"loss_out": LM.metric_for_loss, "metric_out": LM.metric_for_metric}
    model_train.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    # create tf.data.Dataset objects
    tfrgen_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dirname), shuffle=True, epochs=opts.EPOCHS)
    dataset_train = tfrgen_train.get_generator()
    tfrgen_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dirname), shuffle=True, epochs=opts.EPOCHS)
    dataset_val = tfrgen_val.get_generator()
    callbacks = get_callbacks(op.dirname(dst_model))
    steps_per_epoch = count_steps(train_dirname)
    val_steps = np.clip(count_steps(train_dirname)/2, 0, 100).astype(np.int32)

    history = model_train.fit(dataset_train, epochs=opts.EPOCHS, callbacks=callbacks,
                              validation_data=dataset_val, steps_per_epoch=10,
                              validation_steps=val_steps)

    if dst_model:
        dst_model_path = op.join(opts.DATAPATH_CKP, dst_model)
        os.makedirs(op.dirname(dst_model_path), exist_ok=True)
        model_train.save_weights(dst_model_path)

    print("history", history.history.keys())
    # histfile = op.join(model_path, "history.txt")
    # histdata = np.array([history.history["loss"], history.history["acc"],
    #                      history.history["val_loss"], history.history["val_acc"]])
    # np.savetxt(histfile, histdata, fmt="%.3f")
    # print(f"[history]", history)


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


def get_callbacks(model_dir):
    if model_dir is None:
        nowtime = datetime.datetime.now()
        nowtime = nowtime.strftime("%m%d_%H%M%S")
        model_path = op.join(opts.DATAPATH_CKP, nowtime, "model-{epoch:02d}-{val_loss:.2f}.h5")
        log_dir = op.join(opts.DATAPATH_LOG, nowtime)
    else:
        model_path = op.join(opts.DATAPATH_CKP, model_dir, "model-{epoch:02d}-{val_loss:.2f}.h5")
        log_dir = op.join(opts.DATAPATH_LOG, model_dir)

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


def predict():
    pass


def evaluate():
    pass


if __name__ == "__main__":
    train("kitti_raw_train", "kitti_raw_test", "vode_model/model1.h5", "vode_model/model1.h5")

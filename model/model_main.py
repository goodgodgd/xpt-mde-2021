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


def train(train_dirname, val_dirname, model_dir=None):
    stacked_image_shape = (opts.IM_HEIGHT*opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    instrinsic_shape = (3, 3)
    depth_shape = (opts.IM_HEIGHT, opts.IM_WIDTH, 1)
    model_pred, model_train = create_models(stacked_image_shape, instrinsic_shape, depth_shape)

    if model_dir:
        model_train.load_weights(op.join(opts.DATAPATH_CKP, model_dir))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    losses = {"loss": LM.loss_for_loss, "metric": LM.loss_for_metric}
    metrics = {"loss": LM.metric_for_loss, "metric": LM.metric_for_metric}
    model_train.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    # create tf.data.Dataset objects
    tfrgen_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dirname), shuffle=True)
    dataset_train = tfrgen_train.get_generator()
    tfrgen_val = TfrecordGenerator(op.join(opts.DATAPATH_TFR, val_dirname), shuffle=True)
    dataset_val = tfrgen_val.get_generator()
    callbacks = get_callbacks(model_dir)
    steps_per_epoch = count_steps(train_dirname)
    val_steps = np.clip(count_steps(train_dirname), 0, 20)

    history = model_train.fit(dataset_train, epochs=200, callbacks=callbacks,
                              validation_data=dataset_val, steps_per_epoch=steps_per_epoch)


def get_callbacks(model_dir):
    if model_dir is None:
        nowtime = datetime.datetime.now()
        nowtime = nowtime.strftime("%m%d_%H%M%S")
        model_path = op.join(opts.DATAPATH_CKP, nowtime, "model-{epoch:02d}-{val_loss:.2f}.hdf5")
        log_dir = op.join(opts.DATAPATH_LOG, nowtime)
    else:
        model_path = op.join(opts.DATAPATH_CKP, model_dir, "model-{epoch:02d}-{val_loss:.2f}.hdf5")
        log_dir = op.join(opts.DATAPATH_LOG, model_dir)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            period=5
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
    train("kitti_raw_train", "kitti_raw_test")

import os.path as op
import tensorflow as tf

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


def train(train_dirname, test_dirname):
    stacked_image_shape = (opts.IM_HEIGHT*opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    instrinsic_shape = (3, 3)
    depth_shape = (opts.IM_HEIGHT, opts.IM_WIDTH, 1)
    losses = {"loss": LM.loss_for_loss, "metric": LM.loss_for_metric}
    metrics = {"loss": LM.metric_for_loss, "metric": LM.metric_for_metric}

    model_pred, model_train = create_models(stacked_image_shape, instrinsic_shape, depth_shape)
    model_train.compile(optimizer="sgd", loss=losses, metrics=metrics)

    tfrgen_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, train_dirname),
                                     shuffle=True, epochs=10)
    dataset_train = tfrgen_train.get_generator()
    tfrgen_test = TfrecordGenerator(op.join(opts.DATAPATH_TFR, test_dirname))
    dataset_test = tfrgen_test.get_generator()
    history = model_train.fit(dataset_train, validation_data=dataset_test)

    # loss =
    # metric =
    # compile
    # fit


def predict():
    pass


def evaluate():
    pass


if __name__ == "__main__":
    train("kitti_odom_test", "kitti_raw_test")

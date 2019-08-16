import os.path as op
import tensorflow as tf

import settings
from config import opts
from model.model_builder import create_models
from tfrecords.tfrecord_reader import TfrecordGenerator


def train():
    stacked_image_shape = (opts.IM_HEIGHT*opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    instrinsic_shape = (3, 3)
    pose_shape = (opts.SNIPPET_LEN, 6)
    model_pred, model_train = create_models(stacked_image_shape, instrinsic_shape, pose_shape)
    model_train.compile(optimizer="sgd", loss={"vode_loss": loss_for_loss})

    tfrgen_train = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_odom_train"),
                                     shuffle=True, epochs=10)
    dataset_train = tfrgen_train.get_generator()
    tfrgen_test = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_odom_test"))
    dataset_test = tfrgen_test.get_generator()
    history = model_train.fit(dataset_train, validation_data=dataset_test)

    # loss =
    # metric =
    # compile
    # fit


def loss_for_loss(y_true, y_pred):
    return y_pred

def loss_for_metric(y_true, y_pred):
    pass

def predict():
    pass


def evaluate():
    pass


if __name__ == "__main__":
    train()

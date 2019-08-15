import os.path as op

import settings
from config import opts
from model.model_builder import create_models
from tfrecords.tfrecord_reader import TfrecordGenerator


def train():
    stacked_image_shape = (opts.IM_HEIGHT*opts.SNIPPET_LEN, opts.IM_WIDTH, 3)
    model_pred, model_train = create_models(stacked_image_shape, intrin_shape=(3, 3))

    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_odom_train"))
    dataset = tfrgen.get_generator()
    # model.fit(dataset)

    # loss =
    # metric =
    # compile
    # fit


def predict():
    pass


def evaluate():
    pass


if __name__ == "__main__":
    train()

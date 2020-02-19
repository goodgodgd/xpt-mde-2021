import os.path as op
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAvgPool2D

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordGenerator
import gc


def test_tfrecord_reader():
    """
    Test if TfrecordGenerator works fine and print keys and shapes of input tensors
    """
    model = create_model()
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"), shuffle=True)
    dataset = tfrgen.get_generator()
    batch_means = []
    for ei in range(50):
        y_means = []
        for bi, features in enumerate(dataset):
            y = model.predict(features["image"])
            y_means.append(tf.reduce_mean(y, axis=1))
        y_concat = tf.concat(y_means, axis=0).numpy()
        print(f"epoch: {ei}, y_means:", y_concat.shape, y_concat[:5])
        batch_means.append(y_concat)
        """
        garbage collection prevent memory leaking
        BUT it results in memory error in real training...;;
        """
        gc.collect()
    results = np.stack(batch_means, axis=0)
    print("epoch_means:", results.shape, "\n", results)


def create_model():
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(opts.IM_HEIGHT*5, opts.IM_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        GlobalAvgPool2D(),
    ])
    model.compile(optimizer='rmsprop', loss='mean_absolute_error')
    model.summary()
    return model


if __name__ == "__main__":
    test_tfrecord_reader()

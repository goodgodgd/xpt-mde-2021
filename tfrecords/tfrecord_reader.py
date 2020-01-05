import os.path as op
import json
import tensorflow as tf
import cv2

import settings
from config import opts


class TfrecordGenerator:
    def __init__(self, tfrpath, shuffle=False, epochs=1, batch_size=opts.BATCH_SIZE):
        self.tfrpath = tfrpath
        self.shuffle = shuffle
        self.epochs = epochs
        self.batch_size = batch_size
        self.config = self.read_tfrecord_config(tfrpath)
        self.features_dict = self.get_features(self.config)

    def read_tfrecord_config(self, tfrpath):
        with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
            config = json.load(fr)
            for key, feat_conf in config.items():
                if not isinstance(feat_conf, dict):
                    continue
                # convert parse types in string to real type
                if feat_conf["parse_type"] == "tf.string":
                    config[key]["parse_type"] = tf.string
                elif feat_conf["parse_type"] == "tf.int64":
                    config[key]["parse_type"] = tf.int64
                else:
                    raise TypeError("[read_tfrecord_config] invalid parse_type")

                # convert decode types in string to real type
                if feat_conf["decode_type"] == "tf.uint8":
                    config[key]["decode_type"] = tf.uint8
                elif feat_conf["decode_type"] == "tf.float32":
                    config[key]["decode_type"] = tf.float32
                else:
                    raise TypeError("[read_tfrecord_config] invalid decode_type")

        return config

    def get_features(self, config):
        features_dict = {}
        for key, feat_conf in config.items():
            if not isinstance(feat_conf, dict):
                continue
            # convert parse types in string format to real type
            if feat_conf["parse_type"] is tf.string:
                default_value = ""
            elif feat_conf["parse_type"] is tf.int64:
                default_value = 0
            else:
                raise TypeError("[get_features] invalid parse_type")

            features_dict[key] = tf.io.FixedLenFeature((), feat_conf["parse_type"],
                                                       default_value=default_value)
        return features_dict

    def get_generator(self):
        """
        :return features: {"image": .., "pose_gt": .., "depth_gt": .., "intrinsic": ..}
            image: stacked image [batch, height*snippet_len, width, 3]
            pose_gt: 4x4 transformation matrix [batch, num_src, 4, 4]
            depth_gt: gt depth [batch, height, width, 1]
            intrinsic: camera projection matrix [batch, 3, 3]
        """
        file_pattern = f"{self.tfrpath}/*.tfrecord"
        filenames = tf.io.gfile.glob(file_pattern)
        print("[tfrecord reader]", file_pattern, filenames)
        dataset = tf.data.TFRecordDataset(filenames)

        dataset = dataset.map(self.parse_example)
        return self.dataset_process(dataset)

    def parse_example(self, example):
        parsed = tf.io.parse_single_example(example, self.features_dict)
        decoded = {}
        for key, feat_conf in self.config.items():
            if not isinstance(feat_conf, dict):
                continue

            if feat_conf["decode_type"] is None:
                decoded[key] = parsed[key]
            else:
                decoded[key] = tf.io.decode_raw(parsed[key], feat_conf["decode_type"])

            if feat_conf["shape"] is not None:
                decoded[key] = tf.reshape(decoded[key], shape=feat_conf["shape"])

        # raw uint8 type may saturate during bilinear interpolation
        decoded["image"] = tf.image.convert_image_dtype(decoded["image"], dtype=tf.float32) * 2 - 1
        features = {"image": decoded["image"], "pose_gt": decoded["pose"],
                    "depth_gt": decoded["depth"], "intrinsic": decoded["intrinsic"]}
        return features

    def dataset_process(self, dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
            print("[dataset] dataset suffled")
        print(f"[dataset] num epochs={self.epochs}, batch size={self.batch_size}")
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset


# --------------------------------------------------------------------------------
# TESTS

def test_read_dataset():
    """
    Test if TfrecordGenerator works fine and print keys and shapes of input tensors
    """
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"))
    dataset = tfrgen.get_generator()
    for i, x in enumerate(dataset):
        if i == 100:
            break
        print("===== index:", i)
        for key, value in x.items():
            print(f"x shape and type: {key}={value.shape}, {value.dtype}")

        print("gt poses:\n", x['pose_gt'].numpy()[0])
        image = tf.image.convert_image_dtype((x["image"] + 1.)/2., dtype=tf.uint8).numpy()
        cv2.imshow("image", image[0])
        cv2.waitKey(0)


import numpy as np


def test_reuse_dataset():
    """
    Test if generator from TfrecordGenerator can be reused after a full iteration
    """
    np.set_printoptions(precision=3, suppress=True)
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"), epochs=2, batch_size=64)
    dataset = tfrgen.get_generator()
    for i in range(2):
        print("Iteration:", i)
        for k, x in enumerate(dataset):
            print(f"Step: {k}, data: {x['pose_gt'].numpy()[0, 0, 0, :]}")
    print("Dataset is REUSABLE")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    test_read_dataset()
    test_reuse_dataset()

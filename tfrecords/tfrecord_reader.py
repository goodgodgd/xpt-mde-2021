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
        self.feature_dict = self.parse_config(self.config)

    def read_tfrecord_config(self, tfrpath):
        with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
            config = json.load(fr)
            return config

    def parse_config(self, config):
        feature_dict = {}
        for key, feat_conf in config.items():
            # convert parse types in string format to real type
            if feat_conf["parse_type"] == "tf.string":
                config[key]["parse_type"] = tf.string
                default_value = ""
            elif feat_conf["parse_type"] == "tf.int64":
                config[key]["parse_type"] = tf.int64
                default_value = 0
            else:
                raise TypeError()
            # convert decode types in string format to real type
            if feat_conf["decode_type"] == "tf.uint8":
                config[key]["decode_type"] = tf.uint8
            elif feat_conf["decode_type"] == "tf.float32":
                config[key]["decode_type"] = tf.float32
            else:
                config[key]["decode_type"] = None

            feature_dict[key] = tf.io.FixedLenFeature((), config[key]["parse_type"],
                                                      default_value=default_value)
        return feature_dict

    def get_generator(self):
        file_pattern = f"{self.tfrpath}/*.tfrecord"
        filenames = tf.io.gfile.glob(file_pattern)
        print("[tfrecord reader]", file_pattern, filenames)
        dataset = tf.data.TFRecordDataset(filenames)

        dataset = dataset.map(self.parse_example)
        return self.dataset_process(dataset)

    def parse_example(self, example):
        parsed = tf.io.parse_single_example(example, self.feature_dict)
        decoded = {}
        for key, feat_conf in self.config.items():
            if feat_conf["decode_type"] is None:
                decoded[key] = parsed[key]
            else:
                decoded[key] = tf.io.decode_raw(parsed[key], feat_conf["decode_type"])

            if feat_conf["shape"] is not None:
                decoded[key] = tf.reshape(decoded[key], shape=feat_conf["shape"])

        # raw uint8 type may saturate during bilinear interpolation
        decoded["image"] = tf.image.convert_image_dtype(decoded["image"], dtype=tf.float32) * 2 - 1
        x = {"image": decoded["image"], "pose_gt": decoded["pose"],
             "depth_gt": decoded["depth"], "intrinsic": decoded["intrinsic"]}
        # dummy outputs for training
        y = {"loss_out": tf.constant(0, dtype=tf.float32), "metric_out": tf.constant(0, dtype=tf.float32)}
        return x, y

    def dataset_process(self, dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        print(f"===== num epochs={self.epochs}, batchsize={self.batch_size}")
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset


# --------------------------------------------------------------------------------
# TESTS

def test_read_dataset():
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_raw_test"))
    dataset = tfrgen.get_generator()
    for i, (x, y) in enumerate(dataset):
        print("===== index:", i)
        print("x keys:", x.keys())
        print("y data:", y.keys())
        for key, value in x.items():
            print(f"x shape and type: {key}={x[key].shape}, {x[key].dtype}")

        x["image"] = tf.image.convert_image_dtype((x["image"] + 1.)/2., dtype=tf.uint8)
        # x["image"] = tf.cast(x["image"], tf.uint8)
        image = x["image"].numpy()
        cv2.imshow("image", image[0])
        cv2.waitKey(100)


if __name__ == "__main__":
    test_read_dataset()

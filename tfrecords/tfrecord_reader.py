import os.path as op
import json
import tensorflow as tf
import cv2

import settings
from config import opts


class TfrecordGenerator:
    def __init__(self, tfrpath, shuffle=False, epochs=1):
        self.tfrpath = tfrpath
        self.shuffle = shuffle
        self.epochs = epochs
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
        decoded["image"] = tf.cast(decoded["image"], tf.float32)
        x = {"image": decoded["image"], "pose_gt": decoded["pose"],
             "depth_gt": decoded["depth"], "intrinsic": decoded["intrinsic"]}
        y = tf.constant(0)
        return x, y

    # def split_target_and_source(self, image_snippet, pose_snippet):
    #     num_images = opts.SNIPPET_LEN
    #     half_num = int(num_images // 2)
    #     images = []
    #     # slice into list of individual images
    #     for i in range(num_images):
    #         image = tf.slice(image_snippet, [i*opts.IM_HEIGHT, 0, 0], [opts.IM_HEIGHT, -1, -1])
    #         images.append(image)
    #     # split into target and sources
    #     target_image = images.pop(half_num)
    #     source_images = images
    #     source_images = tf.concat(source_images, axis=2)
    #
    #     pose_bef = pose_snippet[:half_num]
    #     pose_aft = pose_snippet[half_num+1:num_images]
    #     source_poses = tf.concat([pose_bef, pose_aft], axis=0)
    #     source_poses = quaternion.from_float_array(source_poses)
    #     source_poses = quaternion.as_rotation_vector(source_poses)
    #
    #     return target_image, source_images, source_poses

    def dataset_process(self, dataset):
        if self.shuffle:
            dataset.shuffle(buffer_size=1000)
        print(f"===== num epochs={self.epochs}, batchsize={opts.BATCH_SIZE}")
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(batch_size=opts.BATCH_SIZE, drop_remainder=True)
        return dataset


# =========

def test():
    tfrgen = TfrecordGenerator(op.join(opts.DATAPATH_TFR, "kitti_odom_test"))
    dataset = tfrgen.get_generator()
    for x, y in dataset:
        print(x.keys())
        print(y)
        for key, value in x.items():
            print(f"x shape and type: {key}={x[key].shape}, {x[key].dtype}")

        x["image"] = tf.cast(x["image"], tf.uint8)
        image = x["image"].numpy()
        cv2.imshow("image", image[0])
        cv2.waitKey(100)


if __name__ == "__main__":
    test()

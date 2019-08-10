import os.path as op
from glob import glob
import tensorflow as tf
import cv2
import numpy as np
import json

import settings
import utils.util_funcs as uf
import tfrecords.data_feeders as df


class TfrecordMaker():
    def __init__(self, srcpath, dstpath):
        self.srcpath = srcpath
        self.dstpath = dstpath
        # check if there is depth data available
        depths = glob(srcpath + "/*/depth/*.npy")
        self.depth_avail = True if depths else False

    def make(self):
        data_feeders = self.create_feeders()
        self.write_tfrecord_config(data_feeders)
        num_images = len(data_feeders["image"])
        num_shards = max(min(num_images // 5000, 10), 1)
        num_images_per_shard = num_images // num_shards
        print(f"========== tfrecord maker started: srcpath={self.srcpath}, dstpath={self.dstpath}")
        print(f"num images={num_images}, shards={num_shards}, images per shard={num_images_per_shard}")
        uf.print_progress(num_images, is_total=True)

        for si in range(num_shards):
            outfile = f"{self.dstpath}/shard_{si:02d}.tfrecord"
            print("\n===== start creating:", outfile)
            with tf.io.TFRecordWriter(outfile) as writer:
                for fi in range(si*num_images_per_shard, (si+1)*num_images_per_shard):
                    uf.print_progress(fi)
                    raw_example = self.create_next_example_dict(data_feeders)
                    serialized = self.make_serialized_example(raw_example)
                    writer.write(serialized)

        print(f"\ntfrecord maker finished: srcpath={self.srcpath}, dstpath={self.dstpath}")

    def create_feeders(self):
        image_files, depth_files, pose_files, intrin_files = self.list_sequence_files()

        def image_reader(filename):
            return cv2.imread(filename)

        def npy_reader(filename):
            data = np.load(filename)
            return data.astype(np.float64)

        def txt_reader(filename):
            data = np.loadtxt(filename)
            return data.astype(np.float64)

        if depth_files:
            depth_feeder = df.NpyFeeder(depth_files, npy_reader)
        else:
            depth_feeder = df.ConstInt64Feeder(0, len(image_files))

        feeders = {"image": df.NpyFeeder(image_files, image_reader),
                   "depth": depth_feeder,
                   "pose": df.NpyFeeder(pose_files, txt_reader),
                   "intrinsic": df.NpyFeeder(intrin_files, txt_reader)
                   }
        return feeders

    def list_sequence_files(self):
        image_files = glob(op.join(self.srcpath, "*/*.png"))
        if self.depth_avail:
            depth_files = [op.join(op.dirname(file_path), "depth", op.basename(file_path))
                           for file_path in image_files]
        else:
            depth_files = []

        pose_files = [op.join(op.dirname(file_path), "pose",
                              op.basename(file_path).replace(".png", ".txt"))
                      for file_path in image_files]
        intrin_files = [op.join(op.dirname(file_path), "intrinsic.txt")
                        for file_path in image_files]

        print("=== list sequence files")
        print(f"frame: {image_files[0:1000:100]}")
        print(f"depth: {depth_files[0:1000:100]}")
        print(f"pose: {pose_files[0:1000:100]}")
        print(f"intrin: {intrin_files[0:1000:100]}")
        
        for files in [image_files, depth_files, pose_files, intrin_files]:
            for file in files:
                assert op.isfile(file), f"{file} NOT exist"

        return image_files, depth_files, pose_files, intrin_files

    def write_tfrecord_config(self, feeders):
        config = dict()
        for key, feeder in feeders.items():
            single_config = {"type": feeder.type, "shape": feeder.shape}
            config[key] = single_config

        print("=== config", config)
        with open(op.join(self.dstpath, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

    @staticmethod
    def create_next_example_dict(feeders):
        example = dict()
        for key, feeder in feeders.items():
            example[key] = feeder.get_next()
        return example

    @staticmethod
    def make_serialized_example(data_dict):
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=data_dict)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=features)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

import os.path as op
from glob import glob
import tensorflow as tf
import json

import settings
import utils.util_funcs as uf
import tfrecords.data_feeders as df


class TfrecordMaker:
    def __init__(self, srcpath, dstpath, stereo, im_width):
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.data_root = op.dirname(op.dirname(dstpath))
        # check if there is depth data available
        depths = glob(srcpath + "/*/depth")
        poses = glob(srcpath + "/*/pose")
        self.depth_avail = True if depths else False
        self.pose_avail = True if poses else False
        self.stereo = stereo
        self.im_width = im_width

    def make(self):
        data_feeders = self.create_feeders()

        self.write_tfrecord_config(data_feeders)
        num_images = len(data_feeders["image"])
        num_shards = max(min(num_images // 2000, 10), 1)
        num_images_per_shard = num_images // num_shards
        print(f"========== tfrecord maker started\n\tsrcpath={self.srcpath}\n\tdstpath={self.dstpath}")
        print(f"\tnum images={num_images}, shards={num_shards}, images per shard={num_images_per_shard}")

        for si in range(num_shards):
            outfile = f"{self.dstpath}/shard_{si:02d}.tfrecord"
            print("\n===== start creating:", outfile.replace(self.data_root, ''))
            with tf.io.TFRecordWriter(outfile) as writer:
                for fi in range(si*num_images_per_shard, (si+1)*num_images_per_shard):
                    raw_example = self.create_next_example_dict(data_feeders)
                    serialized = self.make_serialized_example(raw_example)
                    writer.write(serialized)
                    uf.print_numeric_progress(fi, num_images)

        print(f"\ntfrecord maker finished: srcpath={self.srcpath}, dstpath={self.dstpath}\n")

    def create_feeders(self):
        image_files, intrin_files, depth_files, pose_files, extrin_files = self.list_sequence_files()
        feeders = {"image": df.feeder_factory(image_files, "image", "npyfile", self.stereo),
                   "intrinsic": df.feeder_factory(intrin_files, "intrinsic", "npyfile", self.stereo),
                   }
        if self.depth_avail:
            feeders["depth_gt"] = df.feeder_factory(depth_files, "depth", "npyfile", self.stereo)
        if self.pose_avail:
            feeders["pose_gt"] = df.feeder_factory(pose_files, "pose", "npyfile", self.stereo)

        if self.stereo:
            # add right side data
            feeders_rig = dict()
            for name, left_feeder in feeders.items():
                print("left feeder", name, type(left_feeder))
                feeders_rig[name + "_R"] = df.NpyFileFeederStereoRight(left_feeder)
            feeders.update(feeders_rig)
            # stereo extrinsic
            feeders["stereo_T_LR"] = df.feeder_factory(extrin_files, "extrinsic", "npyfile", self.stereo)

        return feeders

    def list_sequence_files(self):
        image_files = glob(op.join(self.srcpath, "*/*.png"))
        if not image_files:
            raise ValueError(f"[list_sequence_files] no image file in {self.srcpath}")

        intrin_files = [op.join(op.dirname(file_path), "intrinsic.txt")
                        for file_path in image_files]

        if self.depth_avail:
            depth_files = [op.join(op.dirname(file_path), "depth",
                                   op.basename(file_path).replace(".png", ".txt"))
                           for file_path in image_files]
        else:
            depth_files = []

        if self.pose_avail:
            pose_files = [op.join(op.dirname(file_path), "pose",
                                  op.basename(file_path).replace(".png", ".txt"))
                          for file_path in image_files]
        else:
            pose_files = []

        if self.stereo:
            extrin_files = [op.join(op.dirname(file_path), "stereo_T_LR.txt")
                            for file_path in image_files]
        else:
            extrin_files = []

        print("## list sequence files")
        print(f"frame: {[file.replace(self.data_root, '') for file in  image_files[0:1000:200]]}")
        print(f"intrin: {[file.replace(self.data_root, '') for file in  intrin_files[0:1000:200]]}")
        print(f"depth: {[file.replace(self.data_root, '') for file in  depth_files[0:1000:200]]}")
        print(f"pose: {[file.replace(self.data_root, '') for file in  pose_files[0:1000:200]]}")
        print(f"extrin: {[file.replace(self.data_root, '') for file in extrin_files[0:1000:200]]}")

        for files in [image_files, intrin_files, depth_files, pose_files, extrin_files]:
            for file in files:
                assert op.isfile(file), f"{file} NOT exist"

        return image_files, intrin_files, depth_files, pose_files, extrin_files

    def write_tfrecord_config(self, feeders):
        config = dict()
        for key, feeder in feeders.items():
            single_config = {"parse_type": feeder.parse_type, "decode_type": feeder.decode_type, "shape": feeder.shape}
            config[key] = single_config

        config["length"] = len(feeders['image'])
        print("## config", config)
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

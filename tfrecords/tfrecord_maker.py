import os.path as op
from glob import glob
import tensorflow as tf
import shutil
import numpy as np
import json

import settings
import utils.util_funcs as uf
import utils.util_class as uc
from tfrecords.example_maker import ExampleMaker


DATASET_KEYS = {"kitti_raw": ["image", "intrinsic", "depth_gt", "pose_gt",
                              "image_R", "intrinsic_R", "depth_gt_R", "pose_gt_R", "stereo_T_LR"],
                "kitti_odom": ["image", "intrinsic", "pose_gt",
                               "image_R", "intrinsic_R", "pose_gt_R", "stereo_T_LR"],
                "cityscapes": ["image", "intrinsic", "depth_gt", "pose_gt"],
                "waymo": ["image", "intrinsic", "depth_gt", "pose_gt"],
                }


class TfrecordMakerBase:
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        self.dataset = dataset
        self.split = split
        self.srcpath = srcpath
        self.tfrpath = tfrpath
        self.shard_size = shard_size
        self.stereo = stereo
        self.shwc_shape = shwc_shape
        self.example_count = 0
        self.shard_count = 0
        self.drive_paths = self.list_drives(srcpath, split)
        self.data_keys = DATASET_KEYS[dataset]
        self.example_maker = self.get_example_maker(dataset, split, srcpath, stereo, shwc_shape, self.data_keys)
        self.writer = None
        self.pm = uc.PathManager([""])

    def list_drives(self, srcpath, split):
        raise NotImplementedError()

    def get_example_maker(self, *args):
        return ExampleMaker(*args)

    def make(self, max_frames=0):
        stride = self.get_stride(max_frames)
        num_drives = len(self.drive_paths)
        with uc.PathManager(self.tfrpath) as pm:
            self.pm = pm
            for di, drive_path in enumerate(self.drive_paths):
                if self.init_tfrecord(di):
                    continue
                # create data reader in example maker
                self.example_maker.list_frames_(drive_path)
                num_frames = self.example_maker.num_frames()
                for index in range(0, num_frames, stride):
                    example = self.example_maker.get_example(index)
                    example_serial = self.serialize_example(example)
                    if not example_serial:
                        break

                    self.write_tfrecord(example_serial)
                    uf.print_progress_status(f"==[making TFR] drive: {di}/{num_drives}, frame: {index}/{num_frames}")
            self.wrap_up()
            pm.set_ok()

    def get_stride(self, max_frames):
        return 1

    def init_tfrecord(self, drive_index=0):
        raise NotImplementedError()

    def serialize_example(self, example_dict):
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=example_dict)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=features)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

    def write_tfrecord(self, example_serial):
        self.writer.write(example_serial)
        self.example_count += 1
        # reset and create a new tfrecord file
        if self.example_count > self.shard_size:
            self.shard_count += 1
            self.example_count = 0
            self.init_tfrecord()

    def wrap_up(self):
        self.writer.close()


class WaymoTfrecordMaker(TfrecordMakerBase):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def list_drives(self, srcpath, split):
        drive_paths = glob(op.join(srcpath, "training_*"))
        drive_paths.sort()
        return drive_paths

    def init_tfrecord(self, drive_index=0):
        outpath = f"{self.tfrpath}/drive_{drive_index:03d}"
        if op.isdir(outpath):
            print(f"[init_tfrecord] {outpath} exists. move onto the next")
            return True

        # change path to check date integrity
        self.pm.reopen([outpath])
        outfile = f"{outpath}/{self.dataset}_{self.split}_{drive_index:03d}_shard_{self.shard_count:03d}.tfrecord"
        if self.writer is not None:
            self.writer.close()
        self.writer = tf.io.TFRecordWriter(outfile)
        return False

    def wrap_up(self):
        self.writer.close()
        files = glob(f"{self.tfrpath}/*/*.tfrecord")
        for file in files:
            shutil.move(file, op.join(self.tfrpath, op.basename(file)))



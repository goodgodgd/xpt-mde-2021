import os
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
from tfrecords.tfr_util import Serializer, inspect_properties


class TfrecordMakerBase:
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        self.dataset = dataset
        self.split = split
        self.srcpath = srcpath
        self.tfrpath = tfrpath              # final root path of tfrecords of this dataset
        self.tfrpath__ = tfrpath + "__"     # temporary root path of tfrecords of this dataset
        self.tfr_drive_path = tfrpath       # path to write "current" tfrecords
        self.shwc_shape = shwc_shape
        self.shard_size = shard_size        # max number of examples in a shard
        self.shard_count = 0                # number of shards written in this drive
        self.example_count_in_shard = 0     # number of examples in this shard
        self.example_count_in_drive = 0     # number of examples in this drive
        self.drive_paths = self.list_drive_paths(srcpath, split)
        self.data_keys = self.get_dataset_keys(dataset, split, stereo)
        self.example_maker = self.get_example_maker(dataset, split, shwc_shape, self.data_keys)
        self.serialize_example = Serializer()
        self.writer = None
        self.pm = uc.PathManager([""])

    def list_drive_paths(self, srcpath, split):
        raise NotImplementedError()

    def get_dataset_keys(self, dataset, split, stereo):
        keys = []
        if dataset is "kitti_raw":
            keys = ["image", "intrinsic", "depth_gt", "pose_gt"]
            if stereo:
                keys += ["image_R", "intrinsic_R", "depth_gt_R", "pose_gt_R", "stereo_T_LR"]
        elif dataset is "kitti_odom":
            keys = ["image", "intrinsic", "pose_gt"] if split is "test" else ["image", "intrinsic"]
            if stereo:
                if split is "test":
                    keys += ["image_R", "intrinsic_R", "pose_gt_R", "stereo_T_LR"]
                else:
                    keys += ["image_R", "intrinsic_R", "stereo_T_LR"]
        elif dataset is "cityscapes":
            keys = ["image", "intrinsic", "depth_gt", "pose_gt"]
        elif dataset is "waymo":
            keys = ["image", "intrinsic", "depth_gt", "pose_gt"]
        else:
            assert 0, f"[get_dataset_keys] Wrong dataset: {dataset}, {split}, {stereo}"
        return keys

    def get_example_maker(self, *args):
        return ExampleMaker(*args)

    def make(self, max_frames=0):
        num_drives = len(self.drive_paths)
        with uc.PathManager([self.tfrpath__], closer_func=self.on_exit) as pm:
            self.pm = pm
            for di, drive_path in enumerate(self.drive_paths):
                if di > 3:
                    break
                if self.init_tfrecord(di):
                    continue

                # create data reader in example maker
                self.example_maker.init_reader(drive_path)
                num_frames = self.example_maker.num_frames()
                loop_range = self.example_maker.get_range()

                last_example = dict()
                for index in loop_range:
                    example = self.example_maker.get_example(index)
                    if example is None:     # if example was empty, this drive ended
                        break
                    elif not example:       # when dict is empty, skip this index
                        continue
                    example_serial = self.serialize_example(example)
                    if index > 50:
                        break

                    last_example = example
                    self.write_tfrecord(example_serial, di)
                    uf.print_progress_status(f"==[making TFR] drive: {di}/{num_drives}, frame: {index}/{num_frames}")
                self.write_tfrecord_config(last_example)

            pm.set_ok()
        self.wrap_up()

    def init_tfrecord(self, drive_index=0):
        raise NotImplementedError()

    def write_tfrecord(self, example_serial, drive_index):
        self.writer.write(example_serial)
        self.example_count_in_shard += 1
        self.example_count_in_drive += 1
        # reset and create a new tfrecord file
        if self.example_count_in_shard > self.shard_size:
            self.shard_count += 1
            self.example_count_in_shard = 0
            self.open_new_writer(drive_index)

    def open_new_writer(self, drive_index):
        raise NotImplementedError()

    def write_tfrecord_config(self, example):
        raise NotImplementedError()

    def on_exit(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def wrap_up(self):
        raise NotImplementedError()


class WaymoTfrecordMaker(TfrecordMakerBase):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def list_drive_paths(self, srcpath, split):
        # drive_paths = glob(op.join(srcpath, "training_*"))
        drive_paths = []
        if self.split is "train":
            BAD_DRIVES = [2, 4, 5, 10, 11, 12, 17, 25]
        else:
            BAD_DRIVES = [2, 4, 5, 10, 11, 12, 17, 25]
        for di in range(28):
            if di in BAD_DRIVES:
                continue
            drive = op.join(srcpath, f"training_{di:04d}")
            drive_paths.append(drive)
        drive_paths.sort()
        return drive_paths

    def init_tfrecord(self, drive_index=0):
        outpath = f"{self.tfrpath__}/drive_{drive_index:03d}"
        if op.isdir(outpath):
            print(f"[init_tfrecord] {op.basename(outpath)} exists. move onto the next")
            return True

        # change path to check date integrity
        self.pm.reopen([outpath], closer_func=self.on_exit)
        self.tfr_drive_path = outpath
        self.shard_count = 0
        self.example_count_in_shard = 0
        self.example_count_in_drive = 0
        self.open_new_writer(drive_index)
        return False

    def open_new_writer(self, drive_index):
        outfile = f"{self.tfr_drive_path}/{drive_index:03d}_shard_{self.shard_count:03d}.tfrecord"
        self.writer = tf.io.TFRecordWriter(outfile)

    def write_tfrecord_config(self, example):
        config = inspect_properties(example)
        config["length"] = self.example_count_in_drive
        config["imshape"] = self.shwc_shape
        print("## save config", config)
        with open(op.join(self.tfr_drive_path, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

    def wrap_up(self):
        files = glob(f"{self.tfrpath__}/*/*.tfrecord")
        print("[wrap_up] move tfrecords:", files[0:-1:5])
        for file in files:
            shutil.move(file, op.join(self.tfrpath__, op.basename(file)))

        # merge config files of all drives and save only one in tfrpath
        files = glob(f"{self.tfrpath__}/*/tfr_config.txt")
        print("[wrap_up] config files:", files[:5])
        print(f"{self.tfrpath__}/*/config.txt")
        total_length = 0
        config = dict()
        for file in files:
            with open(file, 'r') as fp:
                config = json.load(fp)
                total_length += config["length"]
        config["length"] = total_length
        with open(op.join(self.tfrpath__, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

        os.rename(self.tfrpath__, self.tfrpath)




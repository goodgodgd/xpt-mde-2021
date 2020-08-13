import os
import os.path as op
from glob import glob
import tensorflow as tf
import shutil
import json

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
        self.data_keys = self.get_dataset_keys(dataset.split("__")[0], split, stereo)
        self.example_maker = self.get_example_maker(dataset, split, shwc_shape, self.data_keys)
        self.serialize_example = Serializer()
        self.writer = None
        self.pm = uc.PathManager([""])

    def list_drive_paths(self, srcpath, split):
        raise NotImplementedError()

    def get_dataset_keys(self, dataset, split, stereo):
        keys = []
        if dataset == "kitti_raw":
            keys = ["image", "intrinsic", "depth_gt", "pose_gt"]
            if stereo:
                keys += ["image_R", "intrinsic_R", "depth_gt_R", "pose_gt_R", "stereo_T_LR"]
        elif dataset == "kitti_odom":
            keys = ["image", "intrinsic", "pose_gt"] if split is "test" else ["image", "intrinsic"]
            if stereo:
                if split is "test":
                    keys += ["image_R", "intrinsic_R", "pose_gt_R", "stereo_T_LR"]
                else:
                    keys += ["image_R", "intrinsic_R", "stereo_T_LR"]
        elif dataset == "cityscapes":
            keys = ["image", "intrinsic", "depth_gt", "stereo_T_LR"]
        elif dataset == "waymo":
            keys = ["image", "intrinsic", "depth_gt", "pose_gt"]
        else:
            assert 0, f"[get_dataset_keys] Wrong dataset: {dataset}, {split}, {stereo}"
        return keys

    def get_example_maker(self, dataset, split, shwc_shape, data_keys):
        return ExampleMaker(dataset, split, shwc_shape, data_keys)

    def make(self, max_frames=0):
        num_drives = len(self.drive_paths)
        with uc.PathManager([self.tfrpath__], closer_func=self.on_exit) as pm:
            self.pm = pm
            for di, drive_path in enumerate(self.drive_paths):
                # if di > 3:
                #     break
                if self.init_tfrecord(di):
                    continue

                # create data reader in example maker
                self.example_maker.init_reader(drive_path)
                loop_range = self.example_maker.get_range()
                num_frames = self.example_maker.num_frames()

                last_example = dict()
                for index in loop_range:
                    try:
                        example = self.example_maker.get_example(index)
                    except StopIteration as si: # raised from xxx_reader._get_frame()
                        print("[StopIteration] running drive ended")
                        break
                    except ValueError as ve:    # raised from xxx_reader._get_frame()
                        uf.print_progress_status(f"==[making TFR] ValueError frame: {index}/{num_frames}, {ve}")
                        continue

                    if not example:             # when dict is empty, skip this index
                        uf.print_progress_status(f"==[making TFR] INVALID example, frame: {index}/{num_frames}")
                        continue
                    example_serial = self.serialize_example(example)
                    # if index > 50:
                    #     break

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
        drive_paths = glob(op.join(srcpath, "training_*"))
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
        outfile = f"{self.tfr_drive_path}/drive_{drive_index:03d}_shard_{self.shard_count:03d}.tfrecord"
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


import zipfile


class CityscapesTfrecordMaker(TfrecordMakerBase):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        self.zip_suffix = "extra" if srcpath.endswith("trainextra.zip") else "sequence"
        self.zip_suffix = "sequence" if srcpath.endswith("sequence_trainvaltest.zip") else self.zip_suffix
        print("self.zip_suffix", self.zip_suffix)
        self.zip_files = self.open_zip_files(srcpath)
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def open_zip_files(self, srcpath):
        zip_files = dict()
        zip_files["leftimg"] = zipfile.ZipFile(srcpath, "r")
        if srcpath.endswith("sequence_trainvaltest.zip"):
            zip_files["camera"] = zipfile.ZipFile(srcpath.replace("/leftImg8bit_sequence_", "/camera_"), "r")
        else:
            zip_files["camera"] = zipfile.ZipFile(srcpath.replace("/leftImg8bit_", "/camera_"), "r")
        zip_files["disparity"] = zipfile.ZipFile(srcpath.replace("/leftImg8bit_", "/disparity_"), "r")
        return zip_files

    def get_example_maker(self, dataset, split, shwc_shape, data_keys):
        return ExampleMaker(dataset, split, shwc_shape, data_keys, self.zip_files)

    def list_drive_paths(self, srcpath, split):
        filelist = self.zip_files["leftimg"].namelist()
        filelist = [file for file in filelist if file.endswith(".png")]
        filelist.sort()
        # drive path example: /leftImg8bit_sequence/train/aachen/aachen
        drive_paths = ["_".join(file.split("_")[:-3]) for file in filelist]
        drive_paths = list(set(drive_paths))
        drive_paths.sort()
        return drive_paths

    def init_tfrecord(self, drive_index=0):
        city = self.drive_paths[drive_index].split("/")[-1]
        # example: cityscapes__/sequence_aachen
        outpath = op.join(self.tfrpath__, f"{self.zip_suffix}_{city}")
        print("outpath", outpath)
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
        outfile = f"{self.tfr_drive_path}/{self.zip_suffix}_shard_{self.shard_count:03d}.tfrecord"
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


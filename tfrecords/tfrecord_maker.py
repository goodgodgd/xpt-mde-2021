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
        self.total_example_count = 0        # number of examples in this dataset generated in this session
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
        elif (dataset == "kitti_odom") and (split == "train"):
            keys = ["image", "intrinsic"]
            if stereo:
                keys += ["image_R", "intrinsic_R", "stereo_T_LR"]
        elif (dataset == "kitti_odom") and (split == "test"):
            keys = ["image", "intrinsic", "pose_gt"]
            if stereo:
                keys += ["image_R", "intrinsic_R", "pose_gt_R", "stereo_T_LR"]
        elif dataset == "cityscapes":
            keys = ["image", "intrinsic", "depth_gt"]
            if stereo:
                keys += ["image_R", "intrinsic_R", "stereo_T_LR"]
        elif dataset == "waymo":
            keys = ["image", "intrinsic", "depth_gt", "pose_gt"]
        elif dataset == "driving_stereo":
            keys = ["image", "intrinsic", "depth_gt"]
            if stereo:
                keys += ["image_R", "intrinsic_R", "stereo_T_LR"]
        else:
            assert 0, f"[get_dataset_keys] Wrong dataset: {dataset}, {split}, {stereo}"
        return keys

    def get_example_maker(self, dataset, split, shwc_shape, data_keys):
        return ExampleMaker(dataset, split, shwc_shape, data_keys)

    def make(self, drive_limit=0, frame_limit=0):
        print("\n\n========== Start a new dataset:", op.basename(self.tfrpath))
        num_drives = len(self.drive_paths)
        with uc.PathManager([self.tfrpath__], closer_func=self.on_exit) as pm:
            self.pm = pm
            for di, drive_path in enumerate(self.drive_paths):
                if (drive_limit > 0) and (di >= drive_limit):
                    break
                if self.init_drive_tfrecord(di):
                    continue

                print("\n==== Start a new drive:", drive_path)
                # create data reader in example maker
                self.example_maker.init_reader(drive_path)
                loop_range = self.example_maker.get_range()
                num_frames = self.example_maker.num_frames()

                last_example = dict()
                for ii, index in enumerate(loop_range):
                    try:
                        example = self.example_maker.get_example(index)
                    except StopIteration as si: # raised from xxx_reader._get_frame()
                        print("\n[StopIteration] stop this drive", si)
                        break
                    except ValueError as ve:    # raised from xxx_reader._get_frame()
                        uf.print_progress_status(f"==[making TFR] ValueError frame: {ii}/{num_frames}, {ve}")
                        continue

                    if not example:             # when dict is empty, skip this index
                        uf.print_progress_status(f"==[making TFR] INVALID example, frame: {ii}/{num_frames}")
                        continue
                    example_serial = self.serialize_example(example)
                    if (frame_limit > 0) and (self.example_count_in_drive >= frame_limit):
                        break

                    last_example = example
                    self.write_tfrecord(example_serial, di)
                    uf.print_progress_status(f"==[making TFR] drive: {di}/{num_drives}, "
                                             f"frame: {ii}/{num_frames}, example: {self.example_count_in_drive}, "
                                             f"shard({self.shard_count}): {self.example_count_in_shard}/{self.shard_size}")
                print("")
                self.write_tfrecord_config(last_example)

            pm.set_ok()
        self.wrap_up()

    def init_drive_tfrecord(self, drive_index=0):
        raise NotImplementedError()

    def write_tfrecord(self, example_serial, drive_index):
        self.writer.write(example_serial)
        self.example_count_in_shard += 1
        self.example_count_in_drive += 1
        self.total_example_count += 1
        # reset and create a new tfrecord file
        if self.example_count_in_shard > self.shard_size:
            self.shard_count += 1
            self.example_count_in_shard = 0
            self.open_new_writer(drive_index)

    def open_new_writer(self, drive_index):
        raise NotImplementedError()

    def write_tfrecord_config(self, example):
        config = inspect_properties(example)
        config["length"] = self.example_count_in_drive
        config["imshape"] = self.shwc_shape
        print("## save config", config)
        with open(op.join(self.tfr_drive_path, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

    def on_exit(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def wrap_up(self):
        raise NotImplementedError()

# TODO ======================================================================
# TfrecordMakers which make tfrecords in tfrpath directly


class TfrecordMakerSingleDir(TfrecordMakerBase):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def list_drive_paths(self, srcpath, split):
        raise NotImplementedError()

    def init_drive_tfrecord(self, drive_index=0):
        outpath = self.tfrpath__
        print("[init_drive_tfrecord] outpath:", outpath)
        # change path to check date integrity
        self.pm.reopen([outpath], closer_func=self.on_exit)
        self.tfr_drive_path = outpath
        self.example_count_in_drive = 0
        self.open_new_writer(drive_index)
        return False

    def open_new_writer(self, drive_index):
        outfile = f"{self.tfr_drive_path}/shard_{self.shard_count:03d}.tfrecord"
        self.writer = tf.io.TFRecordWriter(outfile)

    def write_tfrecord_config(self, example):
        config = inspect_properties(example)
        config["length"] = self.total_example_count
        config["imshape"] = self.shwc_shape
        print("## save config", config)
        with open(op.join(self.tfr_drive_path, "tfr_config.txt"), "w") as fr:
            json.dump(config, fr)

    def wrap_up(self):
        os.rename(self.tfrpath__, self.tfrpath)


# For ONLY kitti dataset, tfrecords are generated from extracted files
class KittiRawTfrecordMaker(TfrecordMakerSingleDir):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def get_example_maker(self, dataset, split, shwc_shape, data_keys):
        return ExampleMaker(dataset, split, shwc_shape, data_keys, self.srcpath)

    def list_drive_paths(self, srcpath, split):
        # create drive paths like : ("2011_09_26", "0001")
        split_ = "train" if split == "train" else "test"
        code_tfrecord_path = op.dirname(op.abspath(__file__))
        filename = op.join(code_tfrecord_path, "resources", f"kitti_raw_{split_}_scenes.txt")
        with open(filename, "r") as f:
            drives = f.readlines()
            drives.sort()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            print("[list_drive_paths] drive list:", drives[:5])
        return drives


# For ONLY kitti dataset, tfrecords are generated from extracted files
class KittiOdomTfrecordMaker(TfrecordMakerSingleDir):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)
        self.total_example_count = 0

    def get_example_maker(self, dataset, split, shwc_shape, data_keys):
        return ExampleMaker(dataset, split, shwc_shape, data_keys, self.srcpath)

    def list_drive_paths(self, srcpath, split):
        # create drive paths like : "00"
        if split is "train":
            drives = [f"{i:02d}" for i in range(11, 22)]
            # remove "12" sequence because color distribution is totally different between left and right
            drives.pop(1)
        else:
            drives = [f"{i:02d}" for i in range(0, 11)]
        return drives


class DrivingStereoTfrecordMaker(TfrecordMakerSingleDir):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def list_drive_paths(self, srcpath, split):
        # drive_path like : .../driving_stereo/train-left-image/2018-07-16-15-18-53.zip
        split_ = "train" if split == "train" else "test"
        drive_paths = glob(op.join(srcpath, f"{split_}-left-image", "*.zip"))
        drive_paths.sort()
        return drive_paths

# TODO ======================================================================
# TfrecordMakers which make tfrecords in drive sub-dir under tfrpath and move them to tfrpath when finished


class WaymoTfrecordMaker(TfrecordMakerBase):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)

    def list_drive_paths(self, srcpath, split):
        drive_paths = glob(op.join(srcpath, "training_*"))
        drive_paths.sort()
        return drive_paths

    def init_drive_tfrecord(self, drive_index=0):
        outpath = f"{self.tfrpath__}/drive_{drive_index:03d}"
        print("[init_drive_tfrecord] outpath:", outpath)
        if op.isdir(outpath):
            print(f"[init_drive_tfrecord] {op.basename(outpath)} exists. move onto the next")
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

    def wrap_up(self):
        move_tfrecord_and_merge_configs(self.tfrpath__, self.tfrpath)


import zipfile


class CityscapesTfrecordMaker(TfrecordMakerBase):
    def __init__(self, dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape):
        self.zip_suffix = dataset.split("__")[1]
        self.zip_files = self.open_zip_files(srcpath, self.zip_suffix)
        super().__init__(dataset, split, srcpath, tfrpath, shard_size, stereo, shwc_shape)
        self.city = ""
        print(f"[CityscapesTfrecordMaker] zip_suffix={self.zip_suffix}")

    def open_zip_files(self, srcpath, zip_suffix):
        zip_files = dict()
        if zip_suffix == "extra":
            basic_name = op.join(srcpath, "leftImg8bit_trainextra.zip")
        elif zip_suffix == "sequence":
            basic_name = op.join(srcpath, "leftImg8bit_sequence_trainvaltest.zip")
        else:
            assert 0, f"Wrong zip suffix: {zip_suffix}"

        zip_files["leftImg"] = zipfile.ZipFile(basic_name, "r")
        zip_files["rightImg"] = zipfile.ZipFile(basic_name.replace("/leftImg8bit_", "/rightImg8bit_"), "r")
        if zip_suffix == "extra":
            zip_files["camera"] = zipfile.ZipFile(basic_name.replace("/leftImg8bit_", "/camera_"), "r")
        elif zip_suffix == "sequence":
            zip_files["camera"] = zipfile.ZipFile(basic_name.replace("/leftImg8bit_sequence_", "/camera_"), "r")
        zip_files["disparity"] = zipfile.ZipFile(basic_name.replace("/leftImg8bit_", "/disparity_"), "r")
        return zip_files

    def get_example_maker(self, dataset, split, shwc_shape, data_keys):
        return ExampleMaker(dataset, split, shwc_shape, data_keys, self.zip_files)

    def list_drive_paths(self, srcpath, split):
        filelist = self.zip_files["leftImg"].namelist()
        filelist = [file for file in filelist if file.endswith(".png")]
        filelist.sort()
        # drive path example: /leftImg8bit_sequence/train/aachen/aachen
        drive_paths = ["_".join(file.split("_")[:-3]) for file in filelist]
        drive_paths = list(set(drive_paths))
        drive_paths.sort()
        return drive_paths

    def init_drive_tfrecord(self, drive_index=0):
        city = self.drive_paths[drive_index].split("/")[-1]
        # example: cityscapes__/sequence_aachen
        outpath = op.join(self.tfrpath__, f"{self.zip_suffix}_{city}")
        print("[init_drive_tfrecord] outpath:", outpath)
        if op.isdir(outpath):
            print(f"[init_drive_tfrecord] {op.basename(outpath)} exists. move onto the next")
            return True

        # change path to check date integrity
        self.pm.reopen([outpath], closer_func=self.on_exit)
        self.tfr_drive_path = outpath
        self.city = city
        self.shard_count = 0
        self.example_count_in_shard = 0
        self.example_count_in_drive = 0
        self.open_new_writer(drive_index)
        return False

    def open_new_writer(self, drive_index):
        outfile = f"{self.tfr_drive_path}/{self.zip_suffix}_{self.city}_shard_{self.shard_count:03d}.tfrecord"
        self.writer = tf.io.TFRecordWriter(outfile)

    def wrap_up(self):
        # TODO WARNING!! sequence MUST be created after extra!
        if self.zip_suffix == "sequence":
            move_tfrecord_and_merge_configs(self.tfrpath__, self.tfrpath)


def move_tfrecord_and_merge_configs(tfrpath__, tfrpath):
    files = glob(f"{tfrpath__}/*/*.tfrecord")
    print("[wrap_up] move tfrecords:", files[0:-1:5])
    for file in files:
        shutil.move(file, op.join(tfrpath__, op.basename(file)))

    # merge config files of all drives and save only one in tfrpath
    files = glob(f"{tfrpath__}/*/tfr_config.txt")
    print("[wrap_up] config files:", files[:5])
    total_length = 0
    config = dict()
    for file in files:
        with open(file, 'r') as fp:
            config = json.load(fp)
            total_length += config["length"]
    config["length"] = total_length
    with open(op.join(tfrpath__, "tfr_config.txt"), "w") as fr:
        json.dump(config, fr)

    os.rename(tfrpath__, tfrpath)


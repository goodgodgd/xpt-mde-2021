import os.path as op
import numpy as np
from glob import glob
from PIL import Image
import zipfile

from tfrecords.readers.reader_base import DataReaderBase
from tfrecords.tfr_util import resize_depth_map, apply_color_map


class DrivingStereoReader(DataReaderBase):
    def __init__(self, split=""):
        super().__init__(split)
        self.zip_files = dict()
        self.intrinsic = np.array(0)
        self.intrinsic_R = np.array(0)
        self.stereo_T_LR = np.array(0)

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        prepare variables to read a new sequence data
        drive_path like : .../driving_stereo/train-left-image/2018-07-16-15-18-53.zip
        """
        self.zip_files = self._load_zip_files(drive_path)
        self.frame_names = self.zip_files["leftImg"].namelist()
        self.frame_names.sort()
        calib = self._read_calib(drive_path)
        # TODO check: LEFT is 103 and RIGHT is 101??
        self.intrinsic = np.reshape(calib["P_rect_103"], (3, 4))[:, :3]
        # print("intrinsic:\n", self.intrinsic)
        self.intrinsic_R = np.reshape(calib["P_rect_101"], (3, 4))[:, :3]
        rot = np.reshape(calib["R_103"], (3, 3))
        trn = np.reshape(calib["T_103"], (3, 1))
        T_RL = np.concatenate([np.concatenate([rot, trn], axis=1),
                               np.array([[0, 0, 0, 1]], dtype=np.float32)], axis=0)
        self.stereo_T_LR = np.linalg.inv(T_RL)
        # print("stereo_T_LR:\n", self.stereo_T_LR)

    def _load_zip_files(self, drive_path):
        zip_files = dict()
        left_img_zip = drive_path
        zip_files["leftImg"] = zipfile.ZipFile(left_img_zip)
        right_img_zip = left_img_zip.replace("-left-image", "-right-image")
        zip_files["rightImg"] = zipfile.ZipFile(right_img_zip)
        depth_map_zip = left_img_zip.replace("-left-image", "-depth-map")
        zip_files["depthMap"] = zipfile.ZipFile(depth_map_zip)
        return zip_files

    def _read_calib(self, drive_path):
        calib_file = drive_path.split("/")
        calib_file[-2] = "calib/half-image-calib"
        calib_file = "/".join(calib_file)
        calib_file = calib_file.replace(".zip", ".txt")
        params = dict()
        with open(calib_file, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.rstrip("\n")
                key, values = line.split(":")
                values = values.strip().split(" ")
                values = [float(val) for val in values]
                values = np.array(values, dtype=np.float32)
                params[key] = values
                # print("[_read_calib]", key, values)
        return params

    def num_frames_(self):
        return len(self.frame_names) - 4

    def get_range_(self):
        return range(2, len(self.frame_names)-2)

    def get_image(self, index, right=False):
        filename = self.frame_names[index]
        zipkey = "rightImg" if right else "leftImg"
        image_bytes = self.zip_files[zipkey].open(filename)
        image = Image.open(image_bytes)
        image = np.array(image, np.uint8)
        return image

    def get_pose(self, index, right=False):
        return None

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        assert right is False, "driving stereo dataset has only left depths"
        filename = self.frame_names[index]
        depth_bytes = self.zip_files["depthMap"].open(filename.replace(".jpg", ".png"))
        detph = Image.open(depth_bytes)
        depth = np.array(detph, np.uint16).astype(np.float32) / 256.
        depth = resize_depth_map(depth, srcshape_hw, dstshape_hw)
        return depth.astype(np.float32)

    def get_intrinsic(self, index=0, right=False):
        # loaded in init_drive()
        intrinsic = self.intrinsic_R if right else self.intrinsic
        return intrinsic.copy()

    def get_stereo_extrinsic(self, index=0):
        # loaded in init_drive()
        return self.stereo_T_LR.copy()


import cv2
from config import opts


def test_driving_stereo_reader():
    srcpath = opts.get_raw_data_path("driving_stereo")
    drive_paths = glob(op.join(srcpath, f"train-left-image", "*.zip"))

    for drive_path in drive_paths:
        print("\n!!! New drive start !!!", drive_path)
        reader = DrivingStereoReader("train")
        reader.init_drive(drive_path)
        frame_indices = reader.get_range_()
        for fi in frame_indices:
            image = reader.get_image(fi)
            intrinsic = reader.get_intrinsic(fi)
            depth = reader.get_depth(fi, image.shape[:2], opts.get_shape("HW", "cityscapes"), intrinsic)
            print(f"== test_city_reader) drive: {op.basename(drive_path)}, frame: {fi}")
            view = image
            depth_view = apply_color_map(depth)
            cv2.imshow("image", view)
            cv2.imshow("dstdepth", depth_view)
            key = cv2.waitKey(2000)
            if key == ord('q'):
                break


from tfrecords.tfrecord_reader import TfrecordGenerator
from model.synthesize.synthesize_base import SynthesizeMultiScale
import utils.util_funcs as uf
import utils.convert_pose as cp
import tensorflow as tf


def test_driving_stereo_synthesis():
    tfrpath = op.join(opts.DATAPATH_TFR, "driving_stereo_train")
    dataset = TfrecordGenerator(tfrpath).get_generator()
    batid, srcid = 0, 0

    for i, features in enumerate(dataset):
        if i == 0:
            print("==== check shapes")
            for key, val in features.items():
                print("    ", i, key, val.shape, val.dtype)

        left_target = features["image5d"][:, 4]
        right_source = features["image5d_R"][:, 4:5]    # numsrc = 1
        intrinsic = features["intrinsic"]
        depth_ms = uf.multi_scale_depths(features["depth_gt"], [1, 2, 4, 8])
        pose_r2l = tf.linalg.inv(features["stereo_T_LR"])
        pose_r2l = tf.expand_dims(pose_r2l, axis=1)
        # pose_r2l = tf.tile(pose_r2l, [1, 1, 1, 1])    # numsrc = 1
        pose_r2l = cp.pose_matr2rvec_batch(pose_r2l)
        synth_ms = SynthesizeMultiScale()(right_source, intrinsic, depth_ms, pose_r2l)

        src_image = right_source[batid, srcid]
        tgt_image = left_target[batid]
        syn_image = synth_ms[0][batid, srcid]
        depth_view = apply_color_map(depth_ms[0][batid].numpy())
        view = tf.concat([src_image, tgt_image, syn_image], axis=0)
        view = uf.to_uint8_image(view).numpy()
        view = np.concatenate([view, depth_view], axis=0)
        cv2.imshow("stereo synthesize", view)
        key = cv2.waitKey()
        if key == ord('q'):
            break


if __name__ == "__main__":
    # test_driving_stereo_reader()
    test_driving_stereo_synthesis()


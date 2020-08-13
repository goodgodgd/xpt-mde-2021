import os.path as op
import numpy as np
from glob import glob
from PIL import Image
import json

from tfrecords.readers.reader_base import DataReaderBase


class CityReader(DataReaderBase):
    def __init__(self, split="", reader_arg=None):
        super().__init__(split)
        self.zip_files = reader_arg
        self.camera_names = []
        self.cur_camera_param = dict()
        self.cur_camera_index = -1
        self.target_indices = []

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        prepare variables to read a new sequence data
        """
        self.frame_names = self.zip_files["leftImg"].namelist()
        self.camera_names = self.zip_files["camera"].namelist()
        self.frame_names = [frame for frame in self.frame_names if frame.startswith(drive_path)]
        self.frame_names.sort()

    def num_frames_(self):
        return len(self.target_indices)

    def get_range_(self):
        # list sub drives like /leftImg8bit/train/aachen/aachen/aachen_000000
        sub_drives = ["_".join(frame.split("_")[:-2]) for frame in self.frame_names]
        sub_drives = list(set(sub_drives))
        sub_drives.sort()
        self.target_indices = []
        for sub_drive in sub_drives:
            sub_drive_indices = [fi for fi, frame in enumerate(self.frame_names) if frame.startswith(sub_drive)]
            sub_drive_indices.sort()
            # remove first and last two frames in sub-drives
            sub_drive_indices = sub_drive_indices[2:-2]
            if sub_drive_indices:
                self.target_indices.extend(sub_drive_indices)

        print("[get_range_] target_indices:", self.target_indices[20:40], self.target_indices[50:70])
        return self.target_indices

    def get_image(self, index, right=False):
        # assert right is False, "city dataset is monocular"
        if right:
            filename = self.frame_names[index].replace("leftImg8bit", "rightImg8bit")
            image_bytes = self.zip_files["rightImg"].open(filename)
        else:
            image_bytes = self.zip_files["leftImg"].open(self.frame_names[index])
        image = Image.open(image_bytes)
        image = np.array(image, np.uint8)
        return image

    def get_pose(self, index, right=False):
        return None

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        # assert right is False, "city dataset is monocular"
        params = self._get_camera_param(index)
        baseline = params["extrinsic"]["baseline"]
        fx = params["intrinsic"]["fx"]

        disp_name = self.frame_names[index].replace("leftImg8bit", "disparity")
        # disp_name = disp_name.replace("leftImg8bit", "disparity")
        disp_bytes = self.zip_files["disparity"].open(disp_name)
        disp = Image.open(disp_bytes)
        disp = np.array(disp, np.uint16).astype(np.float32)
        disp[disp > 0] = (disp[disp > 0] - 1) / 256.
        depth = np.zeros(disp.shape, dtype=np.float32)
        depth[disp > 0] = (fx * baseline) / disp[disp > 0]     # depth = baseline * focal length / disparity
        depth = city_resize_depth_map(depth, srcshape_hw, dstshape_hw, intrinsic)
        return depth.astype(np.float32)

    def get_intrinsic(self, index=0, right=False):
        # assert right is False, "city dataset is monocular"
        params = self._get_camera_param(index)
        fx = params["intrinsic"]["fx"]
        fy = params["intrinsic"]["fy"]
        cx = params["intrinsic"]["u0"]
        cy = params["intrinsic"]["v0"]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsic.astype(np.float32)

    def get_stereo_extrinsic(self, index=0):
        params = self._get_camera_param(index)
        baseline = params["extrinsic"]["baseline"]
        # pose to transform points in right frame to left frame
        stereo_T_LR = np.array([[1, 0, 0, baseline], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return stereo_T_LR.astype(np.float32)

    """
    Private methods used inside this class
    """
    def _get_camera_param(self, index):
        if self.cur_camera_index == index:
            return self.cur_camera_param

        filename = self.frame_names[index].replace("leftImg8bit_sequence", "camera")
        filename = filename.replace("leftImg8bit", "camera")
        subdrive = filename.split("_")[:-2]
        subdrive = "_".join(subdrive)
        subdrive_files = [file for file in self.camera_names if file.startswith(subdrive)]
        if not subdrive_files:
            raise ValueError(f"No json file like {subdrive}")

        filename = subdrive_files[0]
        contents = self.zip_files["camera"].read(filename)
        param = json.loads(contents)
        self.cur_camera_param = param
        self.cur_camera_index = index
        return param


def city_resize_depth_map(depth_map, srcshape_hw, dstshape_hw, intrinsic):
    # depth_view = apply_color_map(depth_map)
    # depth_view = cv2.resize(depth_view, (dstshape_hw[1], dstshape_hw[0]))
    # cv2.imshow("srcdepth", depth_view)
    du, dv = np.meshgrid(np.arange(dstshape_hw[1]), np.arange(dstshape_hw[0]))
    du, dv = (du.reshape(-1), dv.reshape(-1))
    scale_y, scale_x = (srcshape_hw[0] / dstshape_hw[0], srcshape_hw[1] / dstshape_hw[1])
    su, sv = (du * scale_x).astype(np.uint16), (dv * scale_y).astype(np.uint16)
    radi_x, radi_y = (int(scale_x/2), int(scale_y/2))
    # print("su", su[0:800:40])
    # print("sv", sv[0:-1:10000])

    dst_depth = np.zeros(du.shape).astype(np.float32)
    weight = np.zeros(du.shape).astype(np.float32)
    for dy in range(-radi_y, radi_y+1):
        for dx in range(-radi_x, radi_x+1):
            v_inds = np.clip(sv + dy, 0, srcshape_hw[0] - 1).astype(np.uint16)
            u_inds = np.clip(su + dx, 0, srcshape_hw[1] - 1).astype(np.uint16)

            # if (dx==1) and (dy==1):
            #     print("u_inds", u_inds[0:400:20])
            #     print("v_inds", v_inds[0:-1:10000])
            tmp_depth = depth_map[v_inds, u_inds]
            tmp_weight = (tmp_depth > 0).astype(np.uint8)
            dst_depth += tmp_depth
            weight += tmp_weight

    dst_depth[weight > 0] /= weight[weight > 0]
    dst_depth = dst_depth.reshape((dstshape_hw[0], dstshape_hw[1], 1))
    return dst_depth


import cv2
from config import opts
import zipfile


def test_city_reader():
    srcpath = "/media/ian/IanBook/datasets/raw_zips/cityscapes/leftImg8bit_sequence_trainvaltest.zip"
    zip_files = dict()
    zip_files["leftImg"] = zipfile.ZipFile(srcpath, "r")
    if srcpath.endswith("sequence_trainvaltest.zip"):
        zip_files["camera"] = zipfile.ZipFile(srcpath.replace("/leftImg8bit_sequence_", "/camera_"), "r")
    else:
        zip_files["camera"] = zipfile.ZipFile(srcpath.replace("/leftImg8bit_", "/camera_"), "r")
    zip_files["disparity"] = zipfile.ZipFile(srcpath.replace("/leftImg8bit_", "/disparity_"), "r")
    drive_paths = list_drive_paths(zip_files["leftImg"].namelist())

    for drive_path in drive_paths:
        print("\n!!! New drive start !!!", drive_path)
        reader = CityReader("train", zip_files)
        reader.init_drive(drive_path)
        frame_indices = reader.get_range_()
        for fi in frame_indices:
            image = reader.get_image(fi)
            intrinsic = reader.get_intrinsic(fi)
            depth = reader.get_depth(fi, image.shape[:2], opts.get_shape("HW", "cityscapes"), intrinsic)
            print(f"== test_city_reader) drive: {op.basename(drive_path)}, frame: {fi}")
            view = image[0:-1:5, 0:-1:5, :]
            depth_view = apply_color_map(depth)
            cv2.imshow("image", view)
            cv2.imshow("dstdepth", depth_view)
            key = cv2.waitKey(2000)
            if key == ord('q'):
                break


def list_drive_paths(filelist):
    filelist = [file for file in filelist if file.endswith(".png")]
    filelist.sort()
    # drive path example: /leftImg8bit_sequence/train/aachen/aachen
    drive_paths = ["_".join(file.split("_")[:-3]) for file in filelist]
    drive_paths = list(set(drive_paths))
    drive_paths.sort()
    return drive_paths


def apply_color_map(depth):
    depth_view = (np.clip(depth, 0, 50.) / 50. * 256).astype(np.uint8)
    depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
    return depth_view


from tfrecords.tfrecord_reader import TfrecordGenerator
from model.synthesize.synthesize_base import SynthesizeMultiScale
import utils.util_funcs as uf
import utils.convert_pose as cp
import tensorflow as tf


def test_city_stereo_synthesis():
    tfrpath = op.join(opts.DATAPATH_TFR, "cityscapes_train")
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
        print("stereo pose:\n", pose_r2l[0, 0].numpy())

        src_image = right_source[batid, srcid]
        tgt_image = left_target[batid]
        syn_image = synth_ms[0][batid, srcid]
        view = tf.concat([src_image, tgt_image, syn_image], axis=0)
        view = uf.to_uint8_image(view).numpy()
        cv2.imshow("stereo synthesize", view)
        cv2.waitKey()


if __name__ == "__main__":
    # test_city_reader()
    test_city_stereo_synthesis()


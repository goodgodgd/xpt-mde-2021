import os
import os.path as op
from glob import glob
import json
import numpy as np
import cv2

from utils.util_class import WrongInputException
from prepare_data.readers.reader_base import DataReaderBase

"""
when 'stereo' is True, 'get_xxx' function returns two data
"""


class CityScapesReader(DataReaderBase):
    def __init__(self, base_path, stereo=False, split=""):
        """
        when 'stereo' is True, 'get_xxx' function returns two data in tuple
        """
        super().__init__(base_path, stereo)
        self.pose_avail = False
        self.depth_avail = True
        self.left_img_dir = "leftImg8bit"
        self.right_img_dir = "rightImg8bit"
        self.split = split

    """
    Public methods used outside this class
    """
    def list_drive_paths(self):
        split_path = op.join(self.base_path, self.left_img_dir, self.split)
        if not op.isdir(split_path):
            raise WrongInputException("[list_sequence_paths] path does NOT exist:" + split_path)

        city_names = os.listdir(split_path)
        city_names = [city for city in city_names if op.isdir(op.join(split_path, city))]
        total_seq_paths = []
        for city in city_names:
            pattern = op.join(split_path, city, "*.png")
            files = glob(pattern)
            seq_numbers = [file.split("_")[-3] for file in files]
            seq_numbers = list(set(seq_numbers))
            seq_numbers.sort()
            # print("sequences:", self.split, city, seq_numbers)
            seq_paths = [op.join(split_path, city, f"{city}_{seq}") for seq in seq_numbers]
            total_seq_paths.extend(seq_paths)

        return total_seq_paths

    def init_drive(self, drive_path):
        """
        reset variables for a new sequence like intrinsic, extrinsic, and last index
        :param drive_path: sequence path like "path/to/cityscapes/leftImg8bit/bochum/bochum_000000"
        :return: number of frames

        self.frame_names: full path of frame files without extension
        """
        self.frame_names = self._find_frame_names(drive_path)
        self.intrinsic = self._find_camera_matrix(0)
        self.T_left_right = self._find_stereo_extrinsic(0)
        return len(self.frame_names)

    # 여기까지 했고 이제 이 아래를 채우면 돼

    def make_saving_paths(self, dstpath, drive_path):
        """
        :param dstpath: path to save reorganized files
        :param drive_path: path to source sequence data
        :return: [image_path, pose_path, depth_path]
                 specific paths under "dstpath" to save image, pose, and depth
        """
        image_path = op.join(dstpath, op.basename(drive_path))
        pose_path = op.join(image_path, "pose") if self.pose_avail else None
        depth_path = op.join(image_path, "depth") if self.depth_avail else None
        return image_path, pose_path, depth_path

    def get_image(self, index):
        frame_name = self.frame_names[index]
        image = self._read_and_crop_image(frame_name, self.left_img_dir)
        if self.stereo:
            frame_name = self.frame_names[index].replace(self.left_img_dir, self.right_img_dir)
            image_rig = self._read_and_crop_image(frame_name, self.right_img_dir)
            return image, image_rig
        else:
            return image

    def get_quat_pose(self, index):
        return None

    def get_depth_map(self, index, raw_img_shape=None, target_shape=None):
        filename = self.frame_names[index]
        filename = filename.replace(self.left_img_dir, "disparity") + "_disparity.png"
        depth = self._read_depth_map(filename, target_shape)
        if self.stereo:
            depth_rig = depth.copy()
            return depth, depth_rig
        return depth

    def get_intrinsic(self):
        return self.intrinsic

    def get_stereo_extrinsic(self):
        return self.T_left_right

    def get_filename(self, index):
        return op.basename(self.frame_names[index])

    """
    Private methods used inside this class
    """
    def _find_frame_names(self, drive_path):
        imgfiles = glob(drive_path + "_*.png")
        # strip extension
        imgfiles = [file.replace(f"_{self.left_img_dir}.png", "") for file in imgfiles]
        imgfiles.sort()
        return imgfiles

    def _read_and_crop_image(self, frame_name, img_dir):
        image_name = frame_name + f"_{img_dir}.png"
        image = cv2.imread(image_name)
        assert image.shape[0] == 1024
        # crop image to remove car body in image
        return image[:768]

    def _find_camera_matrix(self, index):
        intrinsic = self._read_camera_file(index, "intrinsic")
        K = self._make_camera_matrix(intrinsic)
        if self.stereo:
            # TODO WANING!! There is NO calibration results for right side camera
            K_rig = K.copy()
            return K, K_rig
        else:
            return K

    def _read_camera_file(self, index, key, right=False):
        filename = self.frame_names[index].replace(self.left_img_dir, "camera")
        filename = filename + "_camera.json"
        if right:
            filename.replace("left", "right")
        
        with open(filename, "r") as fr:
            camera_params = json.load(fr)
        value = camera_params[key]
        return value

    def _make_camera_matrix(self, intrinsic):
        K = np.zeros((3, 3), dtype=np.float32)
        K[0, 0] = intrinsic["fx"]
        K[1, 1] = intrinsic["fy"]
        K[0, 2] = intrinsic["u0"]
        K[1, 2] = intrinsic["v0"]
        return K

    def _find_stereo_extrinsic(self, index):
        extrinsic = self._read_camera_file(index, "extrinsic")
        tmat = np.identity(4, dtype=np.float32)
        # TODO WARNING!! There is NO extrinsic calibration results for right side camera
        #   the camera frame axes in the Cityscapes dataset = {x: forward, y: left, z: up}
        #   but in this project, we use {x: right, y: down, z: forward}
        #   the right-side camera is in the postive-x-side (right) of the left-side camera
        tmat[0, 3] = extrinsic["baseline"]
        return tmat

    def _read_depth_map(self, filename, target_shape):
        image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        disparity = (image.astype(np.float32) - 1.) / 256.
        depth = np.where(image > 0., disparity, 0)
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]), cv2.INTER_NEAREST)
        return depth

import os.path as op
from glob import glob
import numpy as np

import utils.convert_pose as cp
from prepare_data.readers.reader_base import DataReaderBase

"""
when 'stereo' is True, 'get_xxx' function returns two data
"""


class CityScapesReader(DataReaderBase):
    def __init__(self, base_path, stereo=False, frame_margin=2):
        """
        when 'stereo' is True, 'get_xxx' function returns two data in tuple
        """
        super().__init__(base_path, stereo, frame_margin)
        self.left_img_path = "none"
        self.right_img_path = "none"
        self.split = "none"

    """
    Public methods used outside this class
    """
    def list_drive_paths(self):
        """
        :return: directory paths to sequences of images
        """
        raise NotImplementedError()

    def init_drive(self, drive_path):
        """
        reset variables for a new sequence like intrinsic, extrinsic, and last index
        :param drive_path: sequence drectory path
        :return: frame indices
        """
        raise NotImplementedError()

    def make_saving_paths(self, dstpath, drive_path):
        """
        :param dstpath: path to save reorganized files
        :param drive_path: path to source sequence data
        :return: [image_path, pose_path, depth_path]
                 specific paths under "dstpath" to save image, pose, and depth
        """
        raise NotImplementedError()

    def get_image(self, index):
        """
        :return: indexed image in the current sequence
        """
        raise NotImplementedError()

    def get_intrinsic(self):
        """
        :return: camera projection matrix in the current sequence
        """
        raise NotImplementedError()

    def get_quat_pose(self, index):
        """
        :return: indexed pose in a vector [position, quaternion] in the current sequence
        """
        raise NotImplementedError()

    def get_depth_map(self, index, raw_img_shape, target_shape):
        """
        :return: indexed pose in a vector [position, quaternion] in the current sequence
        """
        raise NotImplementedError()

    def get_stereo_extrinsic(self):
        """
        :return: stereo extrinsic pose that transforms point in right frame into left frame
        """
        raise NotImplementedError()

    """
    Private methods used inside this class
    """

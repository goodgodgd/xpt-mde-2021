import os.path as op
from glob import glob
import numpy as np

import utils.convert_pose as cp

"""
when 'stereo' is True, 'get_xxx' function returns two data
"""


class CityReader:
    def __init__(self, stereo=False):
        self.static_frames = self.read_static_frames()
        self.stereo = stereo
        self.drive_loader = None
        self.pose_avail = False
        self.depth_avail = True
        self.T_left_right = None
        self.frame_count = [0, 0]

    def read_static_frames(self):
        filename = self.static_frame_filename()
        with open(filename, "r") as fr:
            lines = fr.readlines()
            static_frames = [line.strip("\n") for line in lines]
        return static_frames

    def static_frame_filename(self):
        raise NotImplementedError()

    def remove_static_frames(self, frames, drive=""):
        valid_frames = [frame for frame in frames if frame not in self.static_frames]
        return valid_frames

    def list_drive_paths(self, split, base_path):
        raise NotImplementedError()

    def verify_drives(self, drives, base_path):
        # verify drive paths
        verified_drives = []
        for drive in drives:
            drive_path = self.make_raw_data_path(base_path, drive)
            if not op.isdir(drive_path):
                continue
            frame_inds = self.find_frame_indices(drive_path, 5)
            if len(frame_inds) == 0:
                continue
            verified_drives.append(drive)

        print("drive list:", verified_drives)
        print("frame counts:", dict(zip(["total", "non-static"], self.frame_count)))
        return verified_drives

    def make_raw_data_path(self, base_path, drive):
        raise NotImplementedError()

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def find_frame_indices(self, drive_path, snippet_len):
        raise NotImplementedError()

    def find_last_index(self, drive_path):
        frame_files = self.find_frame_files(drive_path)
        last_file = frame_files[-1]
        last_index = last_file.strip("\n").split("/")[-1][:-4]
        return int(last_index)

    def find_frame_files(self, drive_path):
        raise NotImplementedError()

    def get_image(self, index):
        images = self.drive_loader.get_rgb(index)
        if self.stereo:
            return np.array(images[0]), np.array(images[1])
        else:
            return np.array(images[0])

    def get_intrinsic(self):
        intrinsic = self.drive_loader.calib.K_cam2
        if self.stereo:
            intrinsic_rig = self.drive_loader.calib.K_cam3
            return intrinsic, intrinsic_rig
        else:
            return intrinsic

    def get_quat_pose(self, index):
        raise NotImplementedError()

    def get_depth_map(self, frame_idx, drive_path, raw_img_shape, target_shape):
        raise NotImplementedError()

    def get_stereo_extrinsic(self):
        return self.T_left_right

    def update_stereo_extrinsic(self):
        cal = self.drive_loader.calib
        T_cam2_cam3 = np.dot(cal.T_cam2_velo, np.linalg.inv(cal.T_cam3_velo))
        self.T_left_right = T_cam2_cam3
        print("update stereo extrinsic T_left_right =\n", self.T_left_right)


class CitySequenceReader(CityReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)
        self.pose_avail = False
        self.depth_avail = True

    def static_frame_filename(self):
        return op.join(op.dirname(op.abspath(__file__)), "resources", "city_seq_static_frames.txt")

    def list_drive_paths(self, split, base_path):
        filename = op.join(op.dirname(op.abspath(__file__)), "resources", f"kitti_raw_{split}_scenes.txt")
        with open(filename, "r") as f:
            drives = f.readlines()
            drives.sort()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            drives = self.verify_drives(drives, base_path)
            return drives

    def create_drive_loader(self, base_path, drive):
        print(f"[create_drive_loader] pose avail: {self.pose_avail}, depth avail: {self.depth_avail}")

        date, drive_id = drive
        self.drive_loader = pykitti.raw(base_path, date, drive_id)

    def make_raw_data_path(self, base_path, drive):
        drive_path = op.join(base_path, drive[0], f"{drive[0]}_drive_{drive[1]}_sync")
        return drive_path

    def find_frame_indices(self, drive_path, snippet_len):
        raise NotImplementedError()

    def find_frame_files(self, drive_path):
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        frame_files = glob(frame_pattern)
        frame_files.sort()
        return frame_files

    def get_quat_pose(self, index):
        T_w_imu = self.drive_loader.oxts[index].T_w_imu
        T_imu_cam2 = np.linalg.inv(self.drive_loader.calib.T_cam2_imu)
        T_w_cam2 = np.dot(T_w_imu, T_imu_cam2)
        pose_lef = cp.pose_matr2quat(T_w_cam2)
        if self.stereo:
            T_cam2_cam3 = self.T_left_right
            T_w_cam3 = np.dot(T_w_cam2, T_cam2_cam3)
            pose_rig = cp.pose_matr2quat(T_w_cam3)
            return pose_lef, pose_rig
        else:
            return pose_lef

    def get_depth_map(self, frame_idx, drive_path, original_shape, target_shape):
        velo_data = self.drive_loader.get_velo(frame_idx)
        T_cam2_velo, K_cam2 = self.drive_loader.calib.T_cam2_velo, self.drive_loader.calib.K_cam2
        depth = kdg.generate_depth_map(velo_data, T_cam2_velo, K_cam2, original_shape, target_shape)
        if self.stereo:
            T_cam3_velo, K_cam3 = self.drive_loader.calib.T_cam3_velo, self.drive_loader.calib.K_cam3
            depth_rig = kdg.generate_depth_map(velo_data, T_cam3_velo, K_cam3, original_shape, target_shape)
            return depth, depth_rig
        else:
            return depth

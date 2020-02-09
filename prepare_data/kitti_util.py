import os.path as op
from glob import glob
import numpy as np
import pykitti

import prepare_data.kitti_depth_generator as kdg
import utils.convert_pose as cp

"""
when 'stereo' is True, 'get_xxx' function returns two data
"""


class KittiReader:
    def __init__(self, stereo=False):
        self.static_frames = self.read_static_frames()
        self.stereo = stereo
        self.drive_loader = None
        self.pose_avail = True
        self.depth_avail = True
        self.T_left_right = None

    def read_static_frames(self):
        filename = self.static_frame_filename()
        with open(filename, "r") as fr:
            lines = fr.readlines()
            static_frames = [line.strip("\n") for line in lines]
        return static_frames

    def static_frame_filename(self):
        raise NotImplementedError()

    def remove_static_frames(self, frames):
        valid_frames = [frame for frame in frames if frame not in self.static_frames]
        print(f"[remove_static_frames] {len(frames)} -> {len(valid_frames)}")
        return valid_frames

    def list_drives(self, split, base_path):
        raise NotImplementedError()

    def verify_drives(self, drives, base_path):
        # verify drive paths
        verified_drives = []
        for drive in drives:
            drive_path = self.make_drive_path(base_path, drive)
            if not op.isdir(drive_path):
                continue
            frame_inds = self.find_frame_indices(drive_path, 5)
            if len(frame_inds) == 0:
                continue
            verified_drives.append(drive)

        print("drive list:", verified_drives)
        return verified_drives

    def make_drive_path(self, base_path, drive):
        raise NotImplementedError()

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def find_frame_indices(self, drive_path, snippet_len):
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

    def post_proc_indices(self, frame_inds, snippet_len):
        if not frame_inds:
            return frame_inds
        frame_inds.sort()
        last_ind = frame_inds[-1]
        half_len = snippet_len // 2
        frame_inds = [index for index in frame_inds if half_len <= index <= last_ind-half_len]
        frame_inds = np.array(frame_inds, dtype=int)
        print("[find_frame_indices]", frame_inds[:20])
        return frame_inds


class KittiRawReader(KittiReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)
        self.pose_avail = True
        self.depth_avail = True

    def static_frame_filename(self):
        return op.join(op.dirname(op.abspath(__file__)), "resources", "kitti_raw_static_frames.txt")

    def list_drives(self, split, base_path):
        filename = op.join(op.dirname(op.abspath(__file__)), "resources", f"kitti_raw_{split}_scenes.txt")
        with open(filename, "r") as f:
            drives = f.readlines()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            drives = self.verify_drives(drives, base_path)
            return drives

    def create_drive_loader(self, base_path, drive):
        print(f"[create_drive_loader] pose avail: {self.pose_avail}, depth avail: {self.depth_avail}")
        date, drive_id = drive
        self.drive_loader = pykitti.raw(base_path, date, drive_id)

    def make_drive_path(self, base_path, drive):
        drive_path = op.join(base_path, drive[0], f"{drive[0]}_drive_{drive[1]}_sync")
        return drive_path

    def find_frame_indices(self, drive_path, snippet_len):
        raise NotImplementedError()

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


class KittiRawTrainUtil(KittiRawReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)

    def find_frame_indices(self, drive_path, snippet_len):
        # list frame files in drive_path
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        frame_paths = glob(frame_pattern)
        frame_paths.sort()
        frame_files = []
        # reformat to 'date drive_id frame_id' format like '2011_09_26 0001 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files.append(f"{splits[-5]} {splits[-4][-9:-5]} {splits[-1][:-4]}")

        frame_files = self.remove_static_frames(frame_files)
        # convert to frame name to int
        frame_inds = [int(frame.split()[-1]) for frame in frame_files]
        return self.post_proc_indices(frame_inds, snippet_len)


class KittiRawTestUtil(KittiRawReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)

    def find_frame_indices(self, drive_path, snippet_len):
        # count total frames in drive
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        num_frames = len(glob(frame_pattern))
        drive_splits = drive_path.split("/")
        # format drive_path into 'date drive'
        drive_id = f"{drive_splits[-2]} {drive_splits[-1][-9:-5]}"

        filename = op.join(op.dirname(op.abspath(__file__)), "resources", "kitti_test_depth_frames.txt")
        with open(filename, "r") as fr:
            lines = fr.readlines()
            test_frames = [line.strip("\n") for line in lines if line.startswith(drive_id)]
            # remove static frames
            test_frames = self.remove_static_frames(test_frames)
            # convert to int frame indices
            frame_inds = [int(frame.split()[-1]) for frame in test_frames]
            return self.post_proc_indices(frame_inds, snippet_len)


class KittiOdomReader(KittiReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)
        self.poses = []

    def static_frame_filename(self):
        return op.join(op.dirname(op.abspath(__file__)), "resources", "kitti_odom_static_frames.txt")

    def list_drives(self, split, base_path):
        raise NotImplementedError()

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def make_drive_path(self, base_path, drive):
        drive_path = op.join(base_path, "sequences", drive)
        return drive_path

    def find_frame_indices(self, drive_path, snippet_len):
        # list frame files in drive_path
        frame_pattern = op.join(drive_path, "image_2", "*.png")
        frame_paths = glob(frame_pattern)
        frame_paths.sort()
        frame_files = []
        # reformat file paths into 'drive_id frame_id' format like '01 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files.append(f"{splits[-3]} {splits[-1][:-4]}")

        frame_files = self.remove_static_frames(frame_files)
        # convert to frame name to int
        frame_inds = [int(frame.split()[-1]) for frame in frame_files]
        return self.post_proc_indices(frame_inds, snippet_len)

    def get_quat_pose(self, index):
        raise NotImplementedError()

    def get_depth_map(self, frame_idx, drive_path, original_shape, target_shape):
        # no depth available for kitti_odometry dataset
        return None


class KittiOdomTrainReader(KittiOdomReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)
        self.pose_avail = False
        self.depth_avail = False

    def list_drives(self, split, base_path):
        drives = [f"{i:02d}" for i in range(11, 22)]
        drives = self.verify_drives(drives, base_path)
        return drives

    def create_drive_loader(self, base_path, drive):
        print(f"[create_drive_loader] pose avail: {self.pose_avail}, depth avail: {self.depth_avail}")
        self.drive_loader = pykitti.odometry(base_path, drive)

    def get_quat_pose(self, index):
        # no depth available for kitti_odom_train dataset
        return None


class KittiOdomTestReader(KittiOdomReader):
    def __init__(self, stereo=False):
        super().__init__(stereo)
        self.pose_avail = True
        self.depth_avail = False

    def list_drives(self, split, base_path):
        drives = [f"{i:02d}" for i in range(0, 11)]
        drives = self.verify_drives(drives, base_path)
        return drives

    def create_drive_loader(self, base_path, drive):
        pose_file = op.join(base_path, "poses", drive+".txt")
        self.poses = np.loadtxt(pose_file)
        print("read pose file:", pose_file)
        print(f"[create_drive_loader] pose avail: {self.pose_avail}, depth avail: {self.depth_avail}")
        self.drive_loader = pykitti.odometry(base_path, drive)

    def get_quat_pose(self, index):
        T_w_cam2 = self.poses[index].reshape((3, 4))
        T_w_cam2 = np.concatenate([T_w_cam2, np.array([[0, 0, 0, 1]])], axis=0)
        pose = cp.pose_matr2quat(T_w_cam2)
        if self.stereo:
            T_cam2_cam3 = self.T_left_right
            T_w_cam3 = np.dot(T_w_cam2, T_cam2_cam3)
            pose_rig = cp.pose_matr2quat(T_w_cam3)
            return pose, pose_rig
        else:
            return pose


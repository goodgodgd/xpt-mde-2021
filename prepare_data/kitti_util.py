import os.path as op
from glob import glob
import numpy as np
import pykitti
import quaternion

import prepare_data.kitti_depth_generator as kdg
import utils.util_funcs as uf


class KittiUtil:
    def __init__(self):
        self.static_frames = self.read_static_frames()

    def read_static_frames(self):
        filename = self.get_static_frame_file()
        with open(filename, "r") as fr:
            lines = fr.readlines()
            static_frames = [line.strip("\n") for line in lines]
        return static_frames

    def get_static_frame_file(self):
        raise NotImplementedError()

    def remove_static_frames(self, frames):
        valid_frames = [frame for frame in frames if frame not in self.static_frames]
        print(f"[remove_static_frames] {len(frames)} -> {len(valid_frames)}")
        return valid_frames

    def list_drives(self, split):
        raise NotImplementedError()

    def get_drive_path(self, base_path, drive):
        raise NotImplementedError()

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def frame_indices(self, drive_path, snippet_len):
        raise NotImplementedError()

    def get_quat_pose(self, drive_loader, index, drive_path):
        raise NotImplementedError()

    def load_depth_map(self, drive_loader, frame_idx, drive_path, raw_img_shape, target_shape):
        raise NotImplementedError()


class KittiRawUtil(KittiUtil):
    def __init__(self):
        super().__init__()

    def get_static_frame_file(self):
        return op.join(op.dirname(op.abspath(__file__)), "resources",
                       "kitti_raw_static_frames.txt")

    def list_drives(self, split):
        filename = op.join(op.dirname(op.abspath(__file__)), "resources",
                           f"kitti_raw_{split}_scenes.txt")
        with open(filename, "r") as f:
            drives = f.readlines()
            drives = [tuple(drive.strip("\n").split()) for drive in drives]
            print("drive list:", drives)
            return drives

    def create_drive_loader(self, base_path, drive):
        date, drive_id = drive
        return pykitti.raw(base_path, date, drive_id)

    def get_drive_path(self, base_path, drive):
        drive_path = op.join(base_path, drive[0], f"{drive[0]}_drive_{drive[1]}_sync")
        return drive_path

    def frame_indices(self, drive_path, snippet_len):
        raise NotImplementedError()

    def get_quat_pose(self, drive_loader, index, drive_path):
        T_cam2_imu = drive_loader.calib.T_cam2_imu
        T_imu_cam2 = np.linalg.inv(T_cam2_imu)
        T_w_imu = drive_loader.oxts[index].T_w_imu
        T_W_cam2 = np.matmul(T_w_imu, T_imu_cam2)
        return uf.pose_mat2quat(T_W_cam2)

    def load_depth_map(self, drive_loader, frame_idx, drive_path, original_shape, target_shape):
        calib_dir = op.dirname(drive_path)
        velo_data = drive_loader.get_velo(frame_idx)
        depth_map = kdg.generate_depth_map(velo_data, calib_dir, original_shape, target_shape)
        # print(f"depthmap shape={depth_map.shape}, mean={np.mean(depth_map, axis=None)}")
        return depth_map


class KittiRawTrainUtil(KittiRawUtil):
    def __init__(self):
        super().__init__()

    def frame_indices(self, drive_path, snippet_len):
        print("drive path", drive_path)
        # list frame files in drive_path
        frame_pattern = op.join(drive_path, "image_02", "data", "*.png")
        frame_paths = glob(frame_pattern)
        frame_paths.sort()
        frame_paths = frame_paths[snippet_len // 2:-snippet_len // 2]
        frame_files = []
        # reformat to 'date drive_id frame_id' format like '2011_09_26 0001 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files.append(f"{splits[-5]} {splits[-4][-9:-5]} {splits[-1][:-4]}")

        frame_files = self.remove_static_frames(frame_files)
        # convert to frame name to int
        frame_inds = [int(frame.split()[-1]) for frame in frame_files]
        frame_inds.sort()
        frame_inds = np.array(frame_inds, dtype=int)
        print("[frame_indices] frame ids:", frame_inds[0:-1:5])
        return frame_inds


class KittiRawTestUtil(KittiRawUtil):
    def __init__(self):
        super().__init__()

    def frame_indices(self, drive_path, snippet_len):
        print("drive path", drive_path)
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
            frame_inds = [index for index in frame_inds if 2 <= index < num_frames-2]
            frame_inds.sort()
            frame_inds = np.array(frame_inds, dtype=int)
            print("test frames:", frame_inds)
            return frame_inds


class KittiOdomUtil(KittiUtil):
    def __init__(self):
        super().__init__()
        self.poses = []

    def get_static_frame_file(self):
        return op.join(op.dirname(op.abspath(__file__)), "resources",
                       "kitti_odom_static_frames.txt")

    def list_drives(self, split):
        raise NotImplementedError()

    def create_drive_loader(self, base_path, drive):
        raise NotImplementedError()

    def get_drive_path(self, base_path, drive):
        drive_path = op.join(base_path, "sequences", drive)
        return drive_path

    def frame_indices(self, drive_path, snippet_len):
        print("drive path", drive_path)
        # list frame files in drive_path
        frame_pattern = op.join(drive_path, "image_2", "*.png")
        frame_paths = glob(frame_pattern)
        frame_paths.sort()
        frame_paths = frame_paths[snippet_len // 2:-snippet_len // 2]
        frame_files = []
        # reformat file paths into 'drive_id frame_id' format like '01 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files.append(f"{splits[-3]} {splits[-1][:-4]}")

        frame_files = self.remove_static_frames(frame_files)
        # convert to frame name to int
        frame_inds = [int(frame.split()[-1]) for frame in frame_files]
        frame_inds.sort()
        frame_inds = np.array(frame_inds, dtype=int)
        print("[frame_indices] frame ids:", frame_inds[0:-1:10])
        return frame_inds

    def get_quat_pose(self, drive_loader, index, drive_path):
        tmat = self.poses[index].reshape((3, 4))
        return uf.pose_mat2quat(tmat)

    def load_depth_map(self, drive_loader, frame_idx, drive_path, original_shape, target_shape):
        # no depth available for kitti_odom
        return None


class KittiOdomTrainUtil(KittiOdomUtil):
    def __init__(self):
        super().__init__()

    def list_drives(self, split):
        return [f"{i:02d}" for i in range(11, 22)]

    def create_drive_loader(self, base_path, drive):
        loader = pykitti.odometry(base_path, drive)
        nframes = len(loader)
        one_pose = np.concatenate([np.identity(3), np.zeros((3,1))], axis=1).reshape(-1)
        self.poses = np.tile(one_pose, (nframes, 1))
        print("zeros pose:", self.poses.shape)
        return loader


class KittiOdomTestUtil(KittiOdomUtil):
    def __init__(self):
        super().__init__()

    def list_drives(self, split):
        return [f"{i:02d}" for i in range(0, 11)]

    def create_drive_loader(self, base_path, drive):
        pose_file = op.join(base_path, "poses", drive+".txt")
        self.poses = np.loadtxt(pose_file)
        print("read pose file:", pose_file)
        return pykitti.odometry(base_path, drive)

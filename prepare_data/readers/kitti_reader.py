import os.path as op
from glob import glob
import numpy as np
import pykitti

from prepare_data.readers.reader_base import DataReaderBase
import prepare_data.readers.kitti_depth_generator as kdg
import utils.convert_pose as cp


class KittiReader(DataReaderBase):
    def __init__(self, base_path, drive_path, stereo=False):
        """
        when 'stereo' is True, 'get_xxx' function returns two data in tuple
        """
        self.drive_loader = None
        super().__init__(base_path, drive_path, stereo)

    """
    Public methods used outside this class
    """
    def init_drive(self, base_path, drive_path):
        """
        self.frame_names: frame file names without extension
        """
        if (not base_path) or (not drive_path):
            return
        self.drive_loader = self._create_drive_loader(base_path, drive_path)
        self.static_frames = self._read_static_frames()
        self.frame_names, self.frame_indices = self._list_frames(drive_path)
        self.intrinsic = self._find_camera_matrix()
        self.T_left_right = self._find_stereo_extrinsic()

    def num_frames(self):
        return len(self.frame_names)

    def get_image(self, index):
        images = self.drive_loader.get_rgb(index)
        if self.stereo:
            return np.array(images[0]), np.array(images[1])
        else:
            return np.array(images[0])

    def get_intrinsic(self):
        return self.intrinsic

    def get_quat_pose(self, index):
        raise NotImplementedError()

    def get_depth_map(self, index, raw_img_shape=None, target_shape=None):
        raise NotImplementedError()

    def get_stereo_extrinsic(self):
        return self.T_left_right

    def get_filename(self, example_index):
        return self.frame_names[example_index]

    def get_frame_index(self, example_index):
        return self.frame_indices[example_index]

    """
    Private methods used inside this class
    """
    def _read_static_frames(self):
        filename = self._static_frame_filename()
        with open(filename, "r") as fr:
            lines = fr.readlines()
            static_frames = [line.strip("\n") for line in lines]
        return static_frames

    def _static_frame_filename(self):
        raise NotImplementedError()

    def _remove_static_frames(self, frames):
        valid_frames = [frame for frame in frames if frame not in self.static_frames]
        return valid_frames

    def _find_camera_matrix(self):
        intrinsic = self.drive_loader.calib.K_cam2
        if self.stereo:
            intrinsic_rig = self.drive_loader.calib.K_cam3
            return intrinsic, intrinsic_rig
        else:
            return intrinsic

    def _create_drive_loader(self, base_path, drive_path):
        raise NotImplementedError()

    def _list_frames(self, drive_path):
        raise NotImplementedError()

    def _list_frame_files(self, drive_path):
        raise NotImplementedError()

    def _find_stereo_extrinsic(self):
        if self.stereo:
            cal = self.drive_loader.calib
            T_cam2_cam3 = np.dot(cal.T_cam2_velo, np.linalg.inv(cal.T_cam3_velo))
            print("update stereo extrinsic T_left_right =\n", T_cam2_cam3)
            return T_cam2_cam3
        else:
            return None


class KittiRawReader(KittiReader):
    def __init__(self, base_path, drive_path, stereo=False):
        super().__init__(base_path, drive_path, stereo)

    def parse_drive_path(self, drive_path):
        dirsplits = op.basename(drive_path).split("_")
        date = f"{dirsplits[0]}_{dirsplits[1]}_{dirsplits[2]}"
        drive_id = dirsplits[4]
        return date, drive_id

    def _list_frames(self, drive_path):
        raise NotImplementedError()

    def _list_frame_files(self, drive_path):
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

    def get_depth_map(self, index, raw_img_shape=None, target_shape=None):
        velo_data = self.drive_loader.get_velo(index)
        T_cam2_velo, K_cam2 = self.drive_loader.calib.T_cam2_velo, self.drive_loader.calib.K_cam2
        depth = kdg.generate_depth_map(velo_data, T_cam2_velo, K_cam2, raw_img_shape, target_shape)
        if self.stereo:
            T_cam3_velo, K_cam3 = self.drive_loader.calib.T_cam3_velo, self.drive_loader.calib.K_cam3
            depth_rig = kdg.generate_depth_map(velo_data, T_cam3_velo, K_cam3, raw_img_shape, target_shape)
            return depth, depth_rig
        else:
            return depth

    def _static_frame_filename(self):
        prepare_data_path = op.dirname(op.dirname(op.abspath(__file__)))
        return op.join(prepare_data_path, "resources", "kitti_raw_static_frames.txt")

    def _create_drive_loader(self, base_path, drive_path):
        print(f"[_create_drive_loader] drive: {op.basename(drive_path)}")
        date, drive_id = self.parse_drive_path(drive_path)
        return pykitti.raw(base_path, date, drive_id)


class KittiRawTrainReader(KittiRawReader):
    def __init__(self, base_path, drive_path, stereo=False):
        super().__init__(base_path, drive_path, stereo)
        self.split = "train"

    def _list_frames(self, drive_path):
        # list frame files in drive_path
        frame_paths = self._list_frame_files(drive_path)
        frame_files_all = []
        # reformat to 'date drive_id frame_id' format like '2011_09_26 0001 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files_all.append(f"{splits[-5]} {splits[-4][-9:-5]} {splits[-1][:-4]}")

        self.frame_count[0] += len(frame_files_all)
        frame_files = self._remove_static_frames(frame_files_all)
        if len(frame_files) < 2:
            # print(f"[find_frame_names] {op.basename(drive_path)}: {len(frame_files_all)} -> 0")
            return [], []

        self.frame_count[1] += len(frame_files)
        frame_names = [frame.split()[-1] for frame in frame_files]
        frame_names.sort()
        # print(f"[find_frame_names] {op.basename(drive_path)}: {len(frame_files_all)} -> {len(frame_names)}")
        # remove first and last two frames in training drive data
        frame_names = frame_names[2:-2]
        frame_ids = [int(name) for name in frame_names]
        return frame_names, frame_ids


class KittiRawTestReader(KittiRawReader):
    def __init__(self, base_path, drive_path, stereo=False):
        super().__init__(base_path, drive_path, stereo)
        self.split = "test"

    def _list_frames(self, drive_path):
        frame_paths = self._list_frame_files(drive_path)
        drive_splits = drive_path.split("/")
        # format drive_path like 'date drive'
        drive_id = f"{drive_splits[-2]} {drive_splits[-1][-9:-5]}"

        prepare_data_path = op.dirname(op.dirname(op.abspath(__file__)))
        filename = op.join(prepare_data_path, "resources", "kitti_test_depth_frames.txt")
        with open(filename, "r") as fr:
            lines = fr.readlines()
            test_frames = [line.strip("\n") for line in lines if line.startswith(drive_id)]
            self.frame_count[0] += len(test_frames)
            self.frame_count[1] += len(test_frames)
            frame_names = [frame.split()[-1] for frame in test_frames]
            frame_names.sort()
            # print(f"[find_frame_names] {op.basename(drive_path)}: {len(frame_files_all)} -> {len(frame_names)}")
            frame_ids = [int(name) for name in frame_names]
        return frame_names, frame_ids


class KittiOdomReader(KittiReader):
    def __init__(self, base_path, drive_path, stereo=False):
        super().__init__(base_path, drive_path, stereo)
        self.poses = []

    def get_quat_pose(self, index):
        raise NotImplementedError()

    def get_depth_map(self, index, raw_img_shape=None, target_shape=None):
        # no depth available for kitti_odometry dataset
        return None

    def _static_frame_filename(self):
        prepare_data_path = op.dirname(op.dirname(op.abspath(__file__)))
        return op.join(prepare_data_path, "resources", "kitti_odom_static_frames.txt")

    def _create_drive_loader(self, base_path, drive_path):
        raise NotImplementedError()

    def _list_frames(self, drive_path):
        raise NotImplementedError()

    def _list_frame_files(self, drive_path):
        frame_pattern = op.join(drive_path, "image_2", "*.png")
        frame_files = glob(frame_pattern)
        frame_files.sort()
        return frame_files


class KittiOdomTrainReader(KittiOdomReader):
    def __init__(self, base_path, drive_path, stereo=False):
        super().__init__(base_path, drive_path, stereo)

    def get_quat_pose(self, index):
        # no depth available for kitti_odom_train dataset
        return None

    def _create_drive_loader(self, base_path, drive_path):
        drive = op.basename(drive_path)
        print(f"[_create_drive_loader] drive: {drive}")
        return pykitti.odometry(base_path, drive)

    def _list_frames(self, drive_path):
        # list frame files in drive_path
        frame_paths = self._list_frame_files(drive_path)
        frame_files_all = []
        # reformat file paths into 'drive_id frame_id' format like '01 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files_all.append(f"{splits[-3]} {splits[-1][:-4]}")

        self.frame_count[0] += len(frame_files_all)
        frame_files = self._remove_static_frames(frame_files_all)
        if len(frame_files) < 2:
            print(f"[find_frame_names] {op.basename(drive_path)}: {len(frame_files_all)} -> 0")
            return [], []

        self.frame_count[1] += len(frame_files)
        frame_names = [frame.split()[-1] for frame in frame_files]
        frame_names.sort()
        # print(f"[find_frame_names] {op.basename(drive_path)}: {len(frame_files_all)} -> {len(frame_names)}")
        # remove first and last two frames in training drive data
        frame_names = frame_names[2:-2]
        frame_ids = [int(name) for name in frame_names]
        return frame_names, frame_ids


class KittiOdomTestReader(KittiOdomReader):
    def __init__(self, base_path, drive_path, stereo=False):
        super().__init__(base_path, drive_path, stereo)
        self.remove_static = False

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

    def _create_drive_loader(self, base_path, drive_path):
        drive = op.basename(drive_path)
        pose_file = op.join(base_path, "poses", drive+".txt")
        self.poses = np.loadtxt(pose_file)
        print(f"[_create_drive_loader] drive: {drive}")
        return pykitti.odometry(base_path, drive)

    def _list_frames(self, drive_path):
        # list frame files in drive_path
        frame_paths = self._list_frame_files(drive_path)
        frame_files = []
        # reformat file paths into 'drive_id frame_id' format like '01 0000000000'
        for frame in frame_paths:
            splits = frame.strip("\n").split("/")
            frame_files.append(f"{splits[-3]} {splits[-1][:-4]}")

        self.frame_count[0] += len(frame_files)
        self.frame_count[1] += len(frame_files)
        # convert to frame name to int
        frame_names = [int(frame.split()[-1]) for frame in frame_files]
        frame_names.sort()
        # print(f"[find_frame_names] {op.basename(drive_path)}: {len(frame_files)} -> {len(frame_inds)}")
        frame_ids = [int(name) for name in frame_names]
        return frame_names, frame_ids

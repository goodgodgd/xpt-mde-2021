import os
import os.path as op
from glob import glob
import json
import numpy as np
from scipy import sparse
import cv2
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from tfrecords.readers.reader_base import DataReaderBase

T_C2V = tf.constant([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
FRONT_IND = 0


class WaymoReader(DataReaderBase):
    def __init__(self, split=""):
        super().__init__(split)
        self.tfr_dataset = None
        self.frame_buffer = dict()
        self.frame_count = 0

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        prepare variables to read a new sequence data
        """
        self.tfr_dataset = self._get_dataset(drive_path)
        self.tfr_dataset = iter(self.tfr_dataset)
        # self.frame_names =
        # self.intrinsic =
        # self.T_left_right =

    def num_frames(self):
        return 50000

    def get_image(self, index, right=False):
        assert right is False, "waymo dataset is monocular"
        frame = self._get_frame(index)
        front_image = tf.image.decode_jpeg(frame.images[0].image)
        front_image = front_image.numpy()[:, :, [2, 1, 0]]  # rgb to bgr
        return front_image

    def get_pose(self, index, right=False):
        assert right is False, "waymo dataset is monocular"
        frame = self._get_frame(index)
        pose_c2w = tf.reshape(frame.images[0].pose.transform, (4, 4)) @ T_C2V
        return pose_c2w

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        assert right is False, "waymo dataset is monocular"
        return depth

    def get_intrinsic(self, index=0, right=False):
        assert right is False, "waymo dataset is monocular"
        frame = self._get_frame(index)
        intrin = frame.context.camera_calibrations[0].intrinsic
        intrin = np.array([[intrin[0], 0, intrin[2]], [0, intrin[1], intrin[3]], [0, 0, 1]])
        return intrin

    def get_stereo_extrinsic(self):
        return self.T_left_right

    def get_filename(self, example_index):
        filename = op.basename(self.frame_names[example_index])
        return filename.split("_")[-1]

    def list_frames(self, drive_path):
        """
        :param drive_path: sequence path like "train/bochum/bochum_000000"
        :return frame_files: list of frame paths like ["train/bochum/bochum_000000_000001"]
        """
        frame_pattern = self._make_file_path(self.left_img_dir, drive_path + "_*", "png")
        frame_files = glob(frame_pattern)
        split_city = op.dirname(drive_path)
        # list of ["city_seqind_frameid"]
        frame_files = ["_".join(op.basename(file).split("_")[:-1]) for file in frame_files]
        # list of ["split/city/city_seqind_frameid"]
        frame_files = [op.join(split_city, file) for file in frame_files]
        frame_files.sort()
        frame_indices = [int(file.split("_")[-1]) for file in frame_files]
        return frame_files

    """
    Private methods used inside this class
    """
    def _get_dataset(self, drive_path):
        filenames = tf.io.gfile.glob(f"{drive_path}/*.tfrecord")
        print("[tfrecord reader]", drive_path, filenames)
        dataset = tf.data.TFRecordDataset(filenames, compression_type='')
        return dataset

    def _get_frame(self, index):
        if index in self.frame_buffer:
            return self.frame_buffer[index]

        if index == self.frame_count + 1:
            # add a new frame
            frame_data = self.tfr_dataset.__next__()
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))
            self.frame_buffer[index] = frame
            # remove an old frame
            if index - 20 in self.frame_buffer:
                self.frame_buffer.pop(index - 20, None)
            return frame

        assert 0, f"frame index is not consecutive: {self.frame_count} to {index}"

    def _find_camera_matrix(self):
        drive_path = "_".join(self.frame_names[0].split("_")[:-1])
        # e.g. file_pattern = /path/to/cityscapes/camera/train/bochum/bochum_000000_*_camera.png
        file_pattern = self._make_file_path("camera", drive_path + "_*", "json", False)
        json_files = glob(file_pattern)
        assert json_files, "[_find_camera_matrix] There is no json file in pattern: " + file_pattern
        with open(json_files[0], "r") as fr:
            camera_params = json.load(fr)
        intrinsic = camera_params["intrinsic"]
        fx, fy = intrinsic["fx"], intrinsic["fy"]
        cx, cy = intrinsic["u0"], intrinsic["v0"]
        K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]])
        return K

    def _read_depth_map(self, filename, target_shape):
        image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        assert image is not None, f"[_read_depth_map] There is no disparity image " + op.basename(filename)
        disparity = (image.astype(np.float32) - 1.) / 256.
        depth = np.where(image > 0., disparity, 0)
        depth = cv2.resize(depth, (target_shape[1], target_shape[0]), cv2.INTER_NEAREST)
        return depth

    def _make_file_path(self, data_dir, frame_name, extension, add_suffix=True):
        if add_suffix:
            dir_name = data_dir + self.dir_suffix
        else:
            dir_name = data_dir
        return op.join(self.base_path, dir_name, frame_name + f"_{data_dir}.{extension}")


def get_depth_map_manually_project(frame, srcshape_hw, dstshape_hw, intrinsic):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # xyz points in vehicle frame
    points_veh = np.concatenate(points, axis=0)
    # cp_points: (Nx6) [cam_id, ix, iy, cam_id, ix, iy]
    cp_points = np.concatenate(cp_points, axis=0)[:, :3]
    print("points all:", points_veh.shape, "cp_points", cp_points.shape)

    # extract LiDAR points projected to camera[FRONT_IND]
    camera_mask = np.equal(cp_points[:, 0], frame.images[FRONT_IND].name)
    points_veh = points_veh[camera_mask]
    cp_points = cp_points[camera_mask, 1:3]
    print("cam1 points all:", points_veh.shape, "cam1 cp_points", cp_points.shape)

    # transform points from vehicle to camera1
    cam1_T_C2V = tf.reshape(frame.context.camera_calibrations[0].extrinsic.transform, (4, 4)).numpy()
    cam1_T_V2C = np.linalg.inv(cam1_T_C2V)
    points_veh_homo = np.concatenate((points_veh, np.ones((points_veh.shape[0], 1))), axis=1)
    points_veh_homo = points_veh_homo.T
    points_cam_homo = cam1_T_V2C @ points_veh_homo
    points_depth = points_cam_homo[0]

    # project points into image
    # normalize depth to 1
    points_cam = points_cam_homo[:3]
    points_cam_norm = points_cam / points_cam[0:1]
    # scale intrinsic parameters
    scale_y, scale_x = (dstshape_hw[0] / srcshape_hw[0], dstshape_hw[1] / srcshape_hw[1])
    # 3D Y axis = left = -image x,  ix = -Y*fx + cx
    image_x = -points_cam_norm[1] * intrinsic[0, 0] * scale_x + intrinsic[0, 2] * scale_x
    # 3D Z axis = up = -image y,  iy = -Z*fy + cy
    image_y = -points_cam_norm[2] * intrinsic[1, 1] * scale_y + intrinsic[1, 2] * scale_y

    # extract pixels in valid range
    valid_mask = (image_x >= 0) & (image_x <= dstshape_hw[1] - 1) & (image_y >= 0) & (image_y <= dstshape_hw[0] - 1)
    image_x = image_x[valid_mask].astype(np.int32)
    image_y = image_y[valid_mask].astype(np.int32)
    points_depth = points_depth[valid_mask]

    # reconstruct depth map
    depth_map = sparse.coo_matrix((points_depth, (image_y, image_x)), dstshape_hw)
    depth_map = depth_map.toarray()
    return depth_map


def get_depth_map_use_cp(frame, srcshape_hw, dstshape_hw, intrinsic):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    # xyz points in vehicle frame
    points_veh = np.concatenate(points, axis=0)
    # cp_points: (Nx6) [cam_id, ix, iy, cam_id, ix, iy]
    cp_points = np.concatenate(cp_points, axis=0)[:, :3]
    print("points all:", points_veh.shape, "cp_points", cp_points.shape)

    # extract LiDAR points projected to camera[FRONT_IND]
    camera_mask = np.equal(cp_points[:, 0], frame.images[FRONT_IND].name)
    points_veh = points_veh[camera_mask]
    cp_points = cp_points[camera_mask, 1:3]
    print("cam1 points all:", points_veh.shape, "cam1 cp_points", cp_points.shape)

    # transform points from vehicle to camera1
    cam1_T_C2V = tf.reshape(frame.context.camera_calibrations[0].extrinsic.transform, (4, 4)).numpy()
    cam1_T_V2C = np.linalg.inv(cam1_T_C2V)
    points_veh_homo = np.concatenate((points_veh, np.ones((points_veh.shape[0], 1))), axis=1)
    points_veh_homo = points_veh_homo.T
    points_cam_homo = cam1_T_V2C @ points_veh_homo
    points_depth = points_cam_homo[0]

    # scale parameters
    scale_y, scale_x = (dstshape_hw[0] / srcshape_hw[0], dstshape_hw[1] / srcshape_hw[1])
    image_x = cp_points[:, 0] * scale_x
    image_y = cp_points[:, 1] * scale_y
    # extract pixels in valid range
    valid_mask = (image_x >= 0) & (image_x <= dstshape_hw[1] - 1) & (image_y >= 0) & (image_y <= dstshape_hw[0] - 1)
    image_x = image_x[valid_mask].astype(np.int32)
    image_y = image_y[valid_mask].astype(np.int32)
    points_depth = points_depth[valid_mask]

    # reconstruct depth map
    depth_map = sparse.coo_matrix((points_depth, (image_y, image_x)), dstshape_hw)
    depth_map = depth_map.toarray()
    return depth_map




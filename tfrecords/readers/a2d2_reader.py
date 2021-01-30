import os
import zipfile
import tarfile
from PIL import Image
from glob import glob
import os.path as op
import numpy as np
import json
import cv2

from tfrecords.readers.reader_base import DataReaderBase
from tfrecords.tfr_util import resize_depth_map, depth_map_to_point_cloud
from utils.util_funcs import print_progress_status


# TODO: check staic frame 20180810150607_camera_frontcenter_000007746.png


def convert_tar_to_vanilla_zip():
    from config import opts

    print("\n==== convert_tar_to_vanilla_zip")
    tar_pattern = opts.get_raw_data_path("a2d2") + "/../*.tar"
    tar_files = glob(tar_pattern)
    print("tar files:", tar_pattern, tar_files)
    tar_files = [file for file in tar_files if "frontcenter" not in file]

    for ti, tar_name in enumerate(tar_files):
        print("\n====== open tar file:", op.basename(tar_name))
        tfile = tarfile.open(tar_name, 'r')
        filename = op.basename(tar_name).replace(".tar", ".zip")
        zip_name = op.join(op.dirname(tar_name), "zips", filename)
        if op.isfile(zip_name):
            print(f"{op.basename(zip_name)} already made!!")
            continue
        os.makedirs(op.dirname(zip_name), exist_ok=True)
        print("== zip file:", op.basename(zip_name))
        zfile = zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_STORED)

        for fi, tarinfo in enumerate(tfile):
            # if fi >= 100:
            #     break
            if not tarinfo.isfile():
                continue
            inzip_name = tarinfo.name
            contents = tfile.extractfile(tarinfo)
            contents = contents.read()
            zfile.writestr(inzip_name, contents)
            print_progress_status(f"== converting: tar: {ti}, file: {fi}, {inzip_name[-45:]}")

        tfile.close()
        zfile.close()


class A2D2Reader(DataReaderBase):
    def __init__(self, split="", reader_arg=None):
        super().__init__(split)
        self.zip_files = dict()
        self.frame_buffer = dict()
        self.sensor_config = SensorConfig("")
        self.latest_index = 0

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        prepare variables to read a new sequence data
        """
        self.zip_files = self.load_zipfiles(drive_path)
        configfile = op.join(op.dirname(self.zip_files["camera_left"].filename), "cams_lidars.json")
        print("[A2D2Reader] sensor config file:", configfile)
        self.sensor_config = SensorConfig(configfile)
        self.frame_names = self.zip_files["camera_left"].namelist()
        self.frame_names = [name for name in self.frame_names if name.endswith(".png")]
        self.frame_names.sort()

    def load_zipfiles(self, drive_path):
        camera_left = drive_path
        zfiles = dict()
        zfiles["camera_left"] = zipfile.ZipFile(camera_left, "r")
        zfiles["camera_right"] = zipfile.ZipFile(camera_left.replace("camera_frontleft", "camera_frontright"), "r")
        zfiles["lidar_left"] = zipfile.ZipFile(camera_left.replace("camera_frontleft", "lidar_frontleft"), "r")
        zfiles["lidar_right"] = zipfile.ZipFile(camera_left.replace("camera_frontleft", "lidar_frontright"), "r")
        return zfiles

    def num_frames_(self):
        return len(self.frame_names)

    def get_range_(self):
        num_frames = self.num_frames_()
        return range(2, num_frames-2)

    def get_image(self, index, right=False):
        key = "image_R" if right else "image"
        return self.get_frame_data(index, key)

    def get_pose(self, index, right=False):
        return None

    def get_point_cloud(self, index, right=False):
        intrinsic = self.get_intrinsic(index, right)
        key = "depth_gt_R" if right else "depth_gt"
        depth_map = self.get_frame_data(index, key)
        assert (intrinsic.shape == (3, 3)) and (depth_map.ndim == 2), f"[A2D2.get_point_cloud] {intrinsic}, {depth_map.shape}"
        point_cloud = depth_map_to_point_cloud(depth_map, intrinsic)
        return point_cloud

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        key = "depth_gt_R" if right else "depth_gt"
        depth_map = self.get_frame_data(index, key)
        srcshape_hw = self.sensor_config.get_resolution_hw("front_left")
        return resize_depth_map(depth_map, srcshape_hw, dstshape_hw)

    def get_intrinsic(self, index=0, right=False):
        key = "intrinsic_R" if right else "intrinsic"
        return self.get_frame_data(index, key)

    def get_stereo_extrinsic(self, index=0):
        return self.get_frame_data(index, "stereo_T_LR")

    """
    Private methods used inside this class
    """
    def get_frame_data(self, index, key):
        # use loaded frame
        if index in self.frame_buffer:
            return self.frame_buffer[index][key]

        # add new frame
        frame_data = dict()
        frame_data["image"] = self._read_image(index)
        frame_data["intrinsic"] = self.sensor_config.get_cam_matrix("front_left")
        frame_data["depth_gt"] = self._read_depth_map(index)
        frame_data["image_R"] = self._read_image(index, right=True)
        frame_data["intrinsic_R"] = self.sensor_config.get_cam_matrix("front_right")
        frame_data["depth_gt_R"] = self._read_depth_map(index, right=True)
        frame_data["stereo_T_LR"] = self.sensor_config.get_stereo_extrinsic()
        self.frame_buffer[index] = frame_data

        # remove old frames
        if self.latest_index < index:
            self.latest_index = index
        indices_pop = []
        for frame_idx in self.frame_buffer:
            if frame_idx < self.latest_index - 20:
                indices_pop.append(frame_idx)
        for frame_idx in indices_pop:
            self.frame_buffer.pop(frame_idx)

        # return requested data
        return self.frame_buffer[index][key]

    def _read_image(self, index, right=False):
        """
        !! NOTE: images are already undistorted, NO need to undistort images
        """
        if right:
            image_name = self.frame_names[index].replace("frontleft", "frontright").replace("front_left", "front_right")
            zipkey = "camera_right"
            # cam_dir = "front_right"
        else:
            image_name = self.frame_names[index]
            zipkey = "camera_left"
            # cam_dir = "front_left"
        image_bytes = self.zip_files[zipkey].open(image_name)
        image = Image.open(image_bytes)
        image = np.array(image, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = self.sensor_config.undistort_image(image, cam_dir)
        return image

    def _read_depth_map(self, index, right=False):
        image_name = self.frame_names[index]
        if right:
            image_name = image_name.replace("frontleft", "frontright").replace("front_left", "front_right")
        npz_name = image_name.replace("_camera_", "_lidar_").replace("/camera/", "/lidar/").replace(".png", ".npz")
        lidar_key = "lidar_right" if right else "lidar_left"
        npzfile = self.zip_files[lidar_key].open(npz_name)
        npzfile = np.load(npzfile)
        lidar_row = (npzfile["pcloud_attr.row"] + 0.5).astype(np.int32)
        lidar_col = (npzfile["pcloud_attr.col"] + 0.5).astype(np.int32)
        lidar_depth = npzfile["pcloud_attr.depth"]
        camera_key = "front_right" if right else "front_left"
        imsize_hw = self.sensor_config.get_resolution_hw(camera_key)

        assert (lidar_row >= 0).all() and (lidar_row < imsize_hw[0]).all(), \
            f"wrong index: {lidar_row[lidar_row >= 0]}, {lidar_row[lidar_row < imsize_hw[0]]}"
        assert (lidar_col >= 0).all() and (lidar_col < imsize_hw[1]).all(), \
            f"wrong index: {lidar_col[lidar_col >= 0]}, {lidar_col[lidar_col < imsize_hw[1]]}"

        depth_map = np.zeros(imsize_hw, dtype=np.float32)
        depth_map[lidar_row, lidar_col] = lidar_depth
        # depth is supposed to have shape [H, W]
        return depth_map


class SensorConfig:
    # refer to: https://www.a2d2.audi/a2d2/en/tutorial.html
    def __init__(self, cfgfile):
        if cfgfile:
            with open(cfgfile, "r") as fr:
                self.sensor_config = json.load(fr)
        self.undist_remap = dict()

    def get_cam_matrix(self, cam_key):
        intrinsic = np.asarray(self.sensor_config["cameras"][cam_key]["CamMatrix"], dtype=np.float32)
        return intrinsic

    def get_resolution_hw(self, cam_key):
        resolution = self.sensor_config["cameras"][cam_key]["Resolution"]
        resolution = np.asarray([resolution[1], resolution[0]], dtype=np.int32)
        return resolution

    def undistort_image(self, image, cam_name):
        intr_mat_dist = np.asarray(self.sensor_config['cameras'][cam_name]['CamMatrixOriginal'])
        intr_mat_undist = np.asarray(self.sensor_config['cameras'][cam_name]['CamMatrix'])
        dist_parms = np.asarray(self.sensor_config['cameras'][cam_name]['Distortion'])
        lens = self.sensor_config['cameras'][cam_name]['Lens']
        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image

    def __undistort_image_legacy(self, image, cam_name):
        """
        !! NOTE: cv2.remap makes an error when image size is LARGE
        """
        map1, map2 = self._get_undistort_remap(cam_name, (image.shape[1], image.shape[0]))
        print("undistort_image]", image.shape, image.dtype, map1.shape, map1.dtype, map2.shape, map2.dtype)
        print(map1[0:-1:300, 0:-1:300], "\n", map2[0:-1:300, 0:-1:300])
        print("remap min max:", np.min(map1), np.max(map1), np.min(map2), np.max(map2))
        # map1 = np.clip(map1, 0, image.shape[1] - 1)
        # map2 = np.clip(map2, 0, image.shape[0] - 1)
        undist_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        print("undist_image", undist_image.shape, undist_image.dtype)
        return undist_image

    def _get_undistort_remap(self, cam_name, imsize_wh):
        if cam_name in self.undist_remap:
            return self.undist_remap[cam_name]

        # get parameters from config file
        intr_mat_dist = np.asarray(self.sensor_config['cameras'][cam_name]['CamMatrixOriginal'])
        intr_mat_undist = np.asarray(self.sensor_config['cameras'][cam_name]['CamMatrix'])
        dist_parms = np.asarray(self.sensor_config['cameras'][cam_name]['Distortion'])
        lens = self.sensor_config['cameras'][cam_name]['Lens']

        if lens == 'Fisheye':
            remapper = cv2.fisheye.initUndistortRectifyMap(K=intr_mat_dist, D=dist_parms, R=np.eye(3, dtype=np.float32),
                                                           P=intr_mat_undist, size=imsize_wh, m1type=cv2.CV_32FC1)
        elif lens == 'Telecam':
            remapper = cv2.initUndistortRectifyMap(intr_mat_dist, distCoeffs=dist_parms, R=np.eye(3, dtype=np.float32),
                                                   newCameraMatrix=intr_mat_undist, size=imsize_wh, m1type=cv2.CV_32FC1)
        else:
            raise ValueError(f"Wrong camera type {lens}")

        self.undist_remap[cam_name] = remapper
        return remapper

    def get_stereo_extrinsic(self):
        # extrinsic pose: transform points from right frame to left frame
        left_view = self.sensor_config["cameras"]["front_left"]["view"]
        right_view = self.sensor_config["cameras"]["front_right"]["view"]
        return self._transform_right_to_left(left_view, right_view).astype(np.float32)

    def _transform_right_to_left(self, left_view, right_view):
        vehicle_to_left = np.linalg.inv(self._get_transform_to_vehicle(left_view))
        right_to_vehicle = self._get_transform_to_vehicle(right_view)
        transform = np.dot(vehicle_to_left, right_to_vehicle)
        return transform

    def _get_transform_to_vehicle(self, view):
        # get axes (XYZ in sensor config)
        front, left, up = self._get_axes_of_a_view(view)
        # change camera axes from (X:front Y:left) frame to (X:right Y:down) frame
        x_axis, y_axis, z_axis = -left, -up, front
        # get origin
        origin = view['origin']
        transform_to_global = np.eye(4)
        # rotation
        transform_to_global[:3, 0] = x_axis
        transform_to_global[:3, 1] = y_axis
        transform_to_global[:3, 2] = z_axis
        # origin
        transform_to_global[:3, 3] = origin
        # print("transform\n", transform_to_global)
        return transform_to_global

    def _get_axes_of_a_view(self, view):
        x_axis = view['x-axis']
        y_axis = view['y-axis']
        x_axis_norm = np.linalg.norm(x_axis)
        y_axis_norm = np.linalg.norm(y_axis)
        if (x_axis_norm < 1.0e-10) or (y_axis_norm < 1.0e-10):
            raise ValueError("Norm of input vector(s) too small.")
        # normalize the axes
        x_axis = x_axis / x_axis_norm
        y_axis = y_axis / y_axis_norm
        # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
        y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

        # create orthogonal z-axis
        z_axis = np.cross(x_axis, y_axis)
        # calculate and check y-axis and z-axis norms
        y_axis_norm = np.linalg.norm(y_axis)
        z_axis_norm = np.linalg.norm(z_axis)
        if (y_axis_norm < 1.0e-10) or (z_axis_norm < 1.0e-10):
            raise ValueError("Norm of view axis vector(s) too small.")
        # make x/y/z-axes orthonormal
        y_axis = y_axis / y_axis_norm
        z_axis = z_axis / z_axis_norm
        return x_axis, y_axis, z_axis


# ======================================================================
from tfrecords.tfr_util import apply_color_map
from config import opts


def test_read_npz():
    datapath = opts.get_raw_data_path("a2d2")
    filepath = op.join(datapath, "zips", "camera_lidar-20180810150607_lidar_frontright.zip")
    zfile = zipfile.ZipFile(filepath, "r")
    filelist = zfile.namelist()
    npzfile = zfile.open(filelist[5])
    # print("npbytes", len(npbytes), npbytes[:100])
    npzfile = np.load(npzfile)
    print("npz keys:", list(npzfile.keys()))
    point_cloud = npzfile["pcloud_points"]
    print("point_cloud:", point_cloud.shape)
    lidar_id = npzfile["pcloud_attr.lidar_id"]
    print("lidar_id", lidar_id.shape, lidar_id[0:-1:1000])
    lidar_valid = npzfile["pcloud_attr.valid"]
    print("pcloud_attr.valid", lidar_valid.shape, len(lidar_valid[~lidar_valid]))
    lidar_row = npzfile["pcloud_attr.row"]
    print("pcloud_attr.row", lidar_row.shape, np.min(lidar_row), np.max(lidar_row), "\n", lidar_row[0:-1:1000])
    lidar_col = npzfile["pcloud_attr.col"]
    print("pcloud_attr.col", lidar_col.shape, np.min(lidar_col), np.max(lidar_col), "\n", lidar_col[0:-1:1000])
    lidar_depth = npzfile["pcloud_attr.depth"]
    print("pcloud_attr.depth", lidar_depth.shape, np.min(lidar_depth), np.max(lidar_depth), "\n", lidar_depth[0:-1:1000])


def visualize_depth_map():
    datapath = opts.get_raw_data_path("a2d2")
    # read lidar file
    filepath = op.join(datapath, "zips", "camera_lidar-20180810150607_lidar_frontright.zip")
    zfile_lidar = zipfile.ZipFile(filepath, "r")
    lidar_list = zfile_lidar.namelist()
    npzfile = zfile_lidar.open(lidar_list[5])
    npzfile = np.load(npzfile)

    # read image file
    filepath = op.join(datapath, "zips", "camera_lidar-20180810150607_camera_frontright.zip")
    zfile_camera = zipfile.ZipFile(filepath, "r")
    camera_file = lidar_list[5].replace("_lidar_", "_camera_").replace("/lidar/", "/camera/").replace(".npz", ".png")
    image_bytes = zfile_camera.open(camera_file)
    image = Image.open(image_bytes)
    image = np.array(image, np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_view = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    image_crop = image[136:-240]
    image_crop_view = cv2.resize(image_crop, (image_crop.shape[1] // 2, image_crop.shape[0] // 2))

    lidar_row = (npzfile["pcloud_attr.row"] + 0.5).astype(np.int32)
    lidar_col = (npzfile["pcloud_attr.col"] + 0.5).astype(np.int32)
    lidar_depth = npzfile["pcloud_attr.depth"]
    image_shape_hw = (1208, 1920)
    depth_map = np.zeros(image_shape_hw, dtype=np.float32)
    depth_map[lidar_row, lidar_col] = lidar_depth
    depth_color = apply_color_map(depth_map)

    dst_shape_hw = (1208//4, 1920//4)
    print("dst shape:", dst_shape_hw)
    rsz_depth_map = resize_depth_map(depth_map, image_shape_hw, dst_shape_hw)
    rsz_depth_color = apply_color_map(rsz_depth_map)

    image[lidar_row, lidar_col, :] = depth_color[lidar_row, lidar_col, :]
    # cv2.imshow("depthmap", depth_color)
    cv2.imshow("depthrsz", rsz_depth_color)
    cv2.imshow("image", image_view)
    cv2.imshow("image_crop", image_crop_view)
    cv2.waitKey()


def test_a2d2_reader():
    datapath = opts.get_raw_data_path("a2d2")
    imshape_hw = opts.get_img_shape("HW", "a2d2")
    imshape_wh = opts.get_img_shape("WH", "a2d2")
    reader = A2D2Reader("train")
    reader.init_drive(op.join(datapath, "camera_lidar-20180810150607_camera_frontleft.zip"))
    frame_indices = reader.get_range_()
    for index in frame_indices:
        image = reader.get_image(index)
        image_R = reader.get_image(index, right=True)
        intrinsic = reader.get_intrinsic(index)
        extrinsic = reader.get_stereo_extrinsic(index)
        depth = reader.get_depth(index, (0, 0), imshape_hw, intrinsic)
        depth_center = depth[130:140, 200:210]
        depth_center = np.mean(depth_center[depth_center > 0])
        depth[(depth > 20.) & (depth < 23.)] = 0

        print("intrinsic:\n", intrinsic)
        print("extrinsic:\n", extrinsic)
        print("depth center (130:140, 200:210)", depth_center)
        image_view = cv2.resize(image, imshape_wh)
        cv2.imshow("image", image_view)
        image_R_view = cv2.resize(image_R, imshape_wh)
        cv2.imshow("image_R", image_R_view)
        depth_color = apply_color_map(depth)
        cv2.imshow("depth", depth_color)
        cv2.waitKey()


def test_remap():
    """
    cv2.remap function makes error when image size is LARGE
    """
    height, width = 2000, 3000
    map_x, map_y = np.meshgrid(np.arange(0, width, dtype=np.float32), np.arange(0, height, dtype=np.float32))
    map_x -= 2.5
    map_y += 3.5
    image = np.arange(0, map_x.size*3).reshape(height, width, 3).astype(np.uint8)
    stride = width // 10
    print("map_x", map_x.dtype, map_x.shape, "\n", map_x[0:-1:stride, 0:-1:stride])
    print("map_y", map_y.dtype, map_y.shape, "\n", map_y[0:-1:stride, 0:-1:stride])
    print("image", image.dtype, image.shape, "\n", image[0:-1:stride, 0:-1:stride, 0])

    remapped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    print("remapped", remapped.dtype, remapped.shape, "\n", remapped[0:-1:stride, 0:-1:stride, 0])


if __name__ == "__main__":
    convert_tar_to_vanilla_zip()
    # test_read_npz()
    # visualize_depth_map()
    # test_a2d2_reader()
    # test_remap()



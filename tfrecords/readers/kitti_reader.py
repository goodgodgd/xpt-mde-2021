import os.path as op
import numpy as np
from glob import glob
import pykitti
from collections import Counter

from tfrecords.readers.reader_base import DataReaderBase
from tfrecords.tfr_util import apply_color_map
from utils.util_class import MyExceptionToCatch


class KittiRawReader(DataReaderBase):
    def __init__(self, split="", reader_arg=None):
        super().__init__(split)
        self.drive_loader = None
        self.base_path = reader_arg
        self.target_frame_ids = []
        self.intrinsic = np.array(0)
        self.intrinsic_R = np.array(0)
        self.stereo_T_LR = np.array(0)
        self.cur_images = np.array(0)
        self.cur_image_index = -1

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        prepare variables to read a new sequence data
        drive_path example: ("2011_09_26", "0001")
        real path is /media/ian/IanBook/datasets/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync
        """
        drive_key = drive_path
        self.drive_loader = self._create_drive_loader(drive_key)
        self.target_frame_ids = self._list_nonstatic_frame_ids(drive_key)
        self.intrinsic = self._init_intrinsic()
        self.intrinsic_R = self._init_intrinsic(right=True)
        self.stereo_T_LR = self._init_extrinsic()
        print("[KittiRawReader.init_drive] stereo_T_LR:\n", self.stereo_T_LR)

    def num_frames_(self):
        return len(self.target_frame_ids)

    def get_range_(self):
        return self.target_frame_ids

    def get_image(self, index, right=False):
        if self.cur_image_index == index:
            images = self.cur_images
        else:
            images = self.drive_loader.get_rgb(index)
            self.cur_images = images

        image = np.array(images[1]) if right else np.array(images[0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def get_pose(self, index, right=False):
        T_w_imu = self.drive_loader.oxts[index].T_w_imu
        T_imu_cam2 = np.linalg.inv(self.drive_loader.calib.T_cam2_imu)
        T_w_cam2 = np.dot(T_w_imu, T_imu_cam2)
        if right:
            T_cam2_cam3 = self.stereo_T_LR
            T_w_cam3 = np.dot(T_w_cam2, T_cam2_cam3)
            return T_w_cam3.astype(np.float32)
        else:
            return T_w_cam2.astype(np.float32)

    def get_point_cloud(self, index, right=False):
        if index >= len(self.drive_loader.velo_files):
            raise StopIteration("[get_point_cloud] index out of velo_files")
        velo_file = self.drive_loader.velo_files[index]
        velo_index = int(op.basename(velo_file)[:-4])
        if index != velo_index:
            # NOTE: velodyne indices are misaligned with camera indices in sequence ('2011_09_26', '0009')
            # 'index-4' is empirically determined
            index_files = [file for file in self.drive_loader.velo_files if file.endswith(f"{index-4:010d}.bin")]
            print(f"\n---[get_point_cloud] different camera-lidar index:", index, velo_index, op.basename(velo_file), "find:", op.basename(index_files[0]))
            if index_files:
                velo_index = int(op.basename(index_files[0])[:-4])
            else:
                raise MyExceptionToCatch(f"[get_point_cloud] no velodyne file for index {index}")

        velo_in_lidar = self.drive_loader.get_velo(velo_index)
        T2cam = self.drive_loader.calib.T_cam3_velo if right else self.drive_loader.calib.T_cam2_velo
        # velodyne raw data [N, 4] is (forward, left, up, reflectance(0)->1)
        velo_in_lidar[:, 3] = 1
        # points in camera frame [N, 3] (right, down, forward)
        velo_in_camera = np.dot(T2cam, velo_in_lidar.T)
        velo_in_camera = velo_in_camera[:3].T
        # remove all velodyne points behind image plane
        velo_in_camera = velo_in_camera[velo_in_camera[:, 2] > 0]
        return velo_in_camera

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        if index >= len(self.drive_loader.velo_files):
            raise StopIteration("[get_depth] index out of velo_files")
        velo_file = self.drive_loader.velo_files[index]
        velo_index = int(op.basename(velo_file)[:-4])
        if index != velo_index:
            raise StopIteration(f"[get_depth] index does NOT match velo file ID {index} != {velo_index}")

        velo_data = self.drive_loader.get_velo(index)
        if right:
            T_cam3_velo = self.drive_loader.calib.T_cam3_velo
            K_cam3 = self.drive_loader.calib.K_cam3
            depth = generate_depth_map(velo_data, T_cam3_velo, K_cam3, srcshape_hw, dstshape_hw)
        else:
            T_cam2_velo = self.drive_loader.calib.T_cam2_velo
            K_cam2 = self.drive_loader.calib.K_cam2
            depth = generate_depth_map(velo_data, T_cam2_velo, K_cam2, srcshape_hw, dstshape_hw)
        return depth.astype(np.float32)

    def get_intrinsic(self, index=0, right=False):
        # loaded in init_drive()
        intrinsic = self.intrinsic_R if right else self.intrinsic
        return intrinsic.copy().astype(np.float32)

    def get_stereo_extrinsic(self, index=0):
        # loaded in init_drive()
        return self.stereo_T_LR.copy().astype(np.float32)

    """
    Private methods used inside this class
    """
    def _create_drive_loader(self, drive_key):
        # print("confirm drive key : ", drive_key)
        date, drive_id = drive_key
        print("base path : ", self.base_path)
        print("date : ", date)
        print("drive id : ", drive_id)
        return pykitti.raw(self.base_path, date, drive_id)

    def _list_nonstatic_frame_ids(self, drive_key):
        # drive_key example: ("2011_09_26", "0001")
        if self.split != "train":
            return self._read_frame_ids_test(drive_key)

        frame_ids = self._read_frame_ids_train(drive_key)
        # remove first and last two frames that are not appropriate for target frame
        frame_ids = frame_ids[2:-2]
        static_frames = self._read_static_frames()

        date, drive_id = drive_key
        drive_prefix = f"{date} {drive_id}"
        static_frame_ids = [int(frame.split(" ")[-1]) for frame in static_frames if frame.startswith(drive_prefix)]
        frame_ids = set(frame_ids) - set(static_frame_ids)
        frame_ids = list(frame_ids)
        frame_ids.sort()
        return frame_ids

    def _read_frame_ids_test(self, drive_key):
        date, drive_id = drive_key
        drive_prefix = f"{date} {drive_id}"
        prj_tfrecords_path = op.dirname(op.dirname(op.abspath(__file__)))
        filename = op.join(prj_tfrecords_path, "resources", "kitti_test_depth_frames.txt")
        with open(filename, "r") as fr:
            lines = fr.readlines()
            test_frames = [line.strip("\n") for line in lines if line.startswith(drive_prefix)]
            print("[_read_frame_ids_test] test_frames:", len(test_frames), test_frames[:5])
            frame_ids = [int(frame.split()[-1]) for frame in test_frames]
            # frame_ids.sort()

        return frame_ids

    def _read_frame_ids_train(self, drive_key):
        # real path is /media/ian/IanBook/datasets/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync
        date, drive_id = drive_key
        drive_path = op.join(self.base_path, date, f"{date}_drive_{drive_id}_sync", "image_02", "data")
        frames = glob(drive_path + "/*.png")
        # convert file name to integer
        frame_ids = [int(op.basename(frame)[:-4]) for frame in frames]
        return frame_ids

    def _read_static_frames(self):
        prj_tfrecord_path = op.dirname(op.dirname(op.abspath(__file__)))
        filename = op.join(prj_tfrecord_path, "resources", "kitti_raw_static_frames.txt")
        print("static file", filename)
        with open(filename, "r") as fr:
            lines = fr.readlines()
            static_frames = [line.strip("\n") for line in lines]
        return static_frames

    def _init_intrinsic(self, right=False):
        if right:
            return self.drive_loader.calib.K_cam3
        else:
            return self.drive_loader.calib.K_cam2

    def _init_extrinsic(self):
        cal = self.drive_loader.calib
        T_cam2_cam3 = np.dot(cal.T_cam2_velo, np.linalg.inv(cal.T_cam3_velo))
        return T_cam2_cam3


def generate_depth_map(velo_data, T_cam_velo, K_cam, orig_shape, target_shape):
    # remove all velodyne points behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance(0)
    velo_data = velo_data[velo_data[:, 0] >= 0, :].T    # (N, 4) => (4, N)
    velo_data[3, :] = 1
    velo_in_camera = np.dot(T_cam_velo, velo_data)      # => (3, N)

    """ CAUTION!
    orig_shape, target_shape: (height, width) 
    velo_data[i, :] = (x, y, z)
    """
    targ_height, targ_width = target_shape
    orig_height, orig_width = orig_shape
    # rescale intrinsic parameters to target image shape
    K_prj = K_cam.copy()
    K_prj[0, :] *= (targ_width / orig_width)    # fx, cx *= target_width / orig_width
    K_prj[1, :] *= (targ_height / orig_height)  # fy, cy *= target_height / orig_height

    # project the points to the camera
    velo_pts_im = np.dot(K_prj, velo_in_camera[:3])         # => (3, N)
    velo_pts_im[:2] = velo_pts_im[:2] / velo_pts_im[2:3]    # (u, v, z)

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[0] = np.round(velo_pts_im[0]) - 1
    velo_pts_im[1] = np.round(velo_pts_im[1]) - 1
    valid_x = (velo_pts_im[0] >= 0) & (velo_pts_im[0] < targ_width)
    valid_y = (velo_pts_im[1] >= 0) & (velo_pts_im[1] < targ_height)
    velo_pts_im = velo_pts_im[:, valid_x & valid_y]

    # project to image
    depth = np.zeros(target_shape)
    depth[velo_pts_im[1].astype(np.int), velo_pts_im[0].astype(np.int)] = velo_pts_im[2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    depth = depth[:, :, np.newaxis]
    # depth: (height, width, 1)
    return depth


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1

# ======================================================================


class KittiOdomReader(DataReaderBase):
    def __init__(self, split="", reader_arg=None):
        super().__init__(split)
        self.drive_loader = None
        self.base_path = reader_arg
        self.target_frame_ids = []
        self.intrinsic = np.array(0)
        self.intrinsic_R = np.array(0)
        self.poses = np.array(0)
        self.stereo_T_LR = np.array(0)
        self.cur_images = np.array(0)
        self.cur_image_index = -1

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        prepare variables to read a new sequence data
        drive_path example: "00"
        real path is /media/ian/IanBook/datasets/kitti_odometry/sequences/00
        """
        drive_id = drive_path
        drive_path_ = op.join(self.base_path, "sequences", drive_id)
        print("[KittiOdomReader.init_drive] drive_path_:", drive_path_)
        self.drive_loader = self._create_drive_loader(drive_id)
        frame_ids = self._list_frame_ids(drive_path_)
        self.target_frame_ids = frame_ids
        print("[KittiOdomReader.init_drive] frame_ids:", len(frame_ids), frame_ids[:5], frame_ids[-5:])
        if self.split != "train":
            self.poses = self._load_poses(drive_id)     # (N, 4, 4) pose matrices
        self.intrinsic = self._init_intrinsic()
        self.intrinsic_R = self._init_intrinsic(right=True)
        self.stereo_T_LR = self._init_extrinsic()
        print("[KittiRawReader.init_drive] stereo_T_LR:\n", self.stereo_T_LR)

    def num_frames_(self):
        return len(self.target_frame_ids)

    def get_range_(self):
        return self.target_frame_ids

    def get_image(self, index, right=False):
        if self.cur_image_index == index:
            images = self.cur_images
        else:
            images = self.drive_loader.get_rgb(index)
            self.cur_images = images

        image = np.array(images[1]) if right else np.array(images[0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def get_pose(self, index, right=False):
        if self.split == "train":
            return None

        T_w_cam2 = self.poses[index]
        if right:
            T_cam2_cam3 = self.stereo_T_LR
            T_w_cam3 = np.dot(T_w_cam2, T_cam2_cam3)
            return T_w_cam3.astype(np.float32)
        else:
            return T_w_cam2.astype(np.float32)

    def get_point_cloud(self, index, right=False):
        return None

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        return None

    def get_intrinsic(self, index=0, right=False):
        # loaded in init_drive()
        intrinsic = self.intrinsic_R if right else self.intrinsic
        return intrinsic.copy().astype(np.float32)

    def get_stereo_extrinsic(self, index=0):
        # loaded in init_drive()
        return self.stereo_T_LR.copy().astype(np.float32)

    """
    Private methods used inside this class
    """
    def _create_drive_loader(self, drive_id):
        return pykitti.odometry(self.base_path, drive_id)

    def _list_frame_ids(self, drive_path):
        image_pattern = op.join(drive_path, "image_2", "*.png")
        frames = glob(image_pattern)
        # convert file name to integer
        frame_ids = [int(op.basename(frame)[:-4]) for frame in frames]
        # remove first and last two frames that are not appropriate for target frame
        if self.split == "train":
            frame_ids = frame_ids[2:-2]
        frame_ids.sort()
        return frame_ids

    def _load_poses(self, drive_id):
        pose_file = op.join(self.base_path, "poses", drive_id + ".txt")
        poses = np.loadtxt(pose_file)
        homogeneous = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), reps=(poses.shape[0], 1))
        poses = np.concatenate([poses, homogeneous], axis=1)
        poses = np.reshape(poses, (poses.shape[0], 4, 4))
        return poses

    def _init_intrinsic(self, right=False):
        if right:
            return self.drive_loader.calib.K_cam3
        else:
            return self.drive_loader.calib.K_cam2

    def _init_extrinsic(self):
        cal = self.drive_loader.calib
        T_cam2_cam3 = np.dot(cal.T_cam2_velo, np.linalg.inv(cal.T_cam3_velo))
        return T_cam2_cam3


# ======================================================================
import cv2
from config import opts


def test_kitti_raw_reader():
    print("\n===== start test_kitti_raw_reader")
    srcpath = opts.get_raw_data_path("kitti_raw")
    drive_keys = [("2011_09_26", "0011"), ("2011_09_26", "0017")]

    for drive_key in drive_keys:
        print("\n!!! New drive start !!!", drive_key)
        reader = KittiRawReader("train", srcpath)
        reader.init_drive(drive_key)
        frame_indices = reader.get_range_()
        for fi in frame_indices:
            image = reader.get_image(fi)
            image_R = reader.get_image(fi, right=True)
            intrinsic = reader.get_intrinsic(fi)
            extrinsic = reader.get_stereo_extrinsic(fi)
            depth = reader.get_depth(fi, image.shape[:2], opts.get_img_shape("HW", "kitti_raw"), intrinsic)
            print(f"== test_kitti_raw_reader) drive: {drive_key}, frame id: {fi}")
            print("image size:", image.shape)
            print("intrinsic:\n", intrinsic)
            print("extrinsic:\n", extrinsic)
            image_LR = np.concatenate([image, image_R], axis=0)
            cv2.imshow("image", image_LR)
            depth_view = apply_color_map(depth)
            cv2.imshow("dstdepth", depth_view)
            key = cv2.waitKey(2000)
            if key == ord('q'):
                break


def test_kitti_odom_reader():
    print("\n===== start test_kitti_odom_reader")
    srcpath = opts.get_raw_data_path("kitti_odom")
    drive_keys = ["00", "10"]

    for drive_id in drive_keys:
        print("\n!!! New drive start !!!", drive_id)
        reader = KittiOdomReader("train", srcpath)
        reader.init_drive(drive_id)
        frame_indices = reader.get_range_()
        print("frame_indices", frame_indices[:5], frame_indices[-5:])
        for ii, fi in enumerate(frame_indices):
            image = reader.get_image(fi)
            image_R = reader.get_image(fi, right=True)
            intrinsic = reader.get_intrinsic(fi)
            pose = reader.get_pose(fi)
            extrinsic = reader.get_stereo_extrinsic(fi)
            if ii == 0:
                print("image size:", image.shape)
                print("intrinsic:\n", intrinsic)
                print("extrinsic:\n", extrinsic)

            print(f"== test_kitti_odom_reader) drive: {drive_id}, frame id: {fi}")
            print("pose\n", pose)
            image_LR = np.concatenate([image, image_R], axis=0)
            cv2.imshow("image", image_LR)
            key = cv2.waitKey(2000)
            if key == ord('q'):
                break


from tfrecords.tfrecord_reader import TfrecordReader
from model.synthesize.synthesize_base import SynthesizeMultiScale
import utils.util_funcs as uf
import utils.convert_pose as cp
import tensorflow as tf


def test_kitti_raw_synthesis():
    print("\n===== start test_kitti_raw_synthesis")
    tfrpath = op.join(opts.DATAPATH_TFR, "kitti_raw_train")
    dataset = TfrecordReader(tfrpath).get_dataset()
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


def test_crop_image():
    print("\n===== start test_crop_image")
    srcpath = opts.get_raw_data_path("kitti_raw")
    drive_keys = [("2011_09_26", "0011"), ("2011_09_26", "0017")]

    for drive_key in drive_keys:
        print("\n!!! New drive start !!!", drive_key)
        reader = KittiRawReader("train", srcpath)
        reader.init_drive(drive_key)
        frame_indices = reader.get_range_()
        for fi in frame_indices:
            image = reader.get_image(fi)
            height, width = image.shape[:2]
            vcrop_size = (width // 4, width)
            hcrop_size = (height, height*3)

            row_beg = int((height - vcrop_size[0]) * 0.6)
            vcrop_middle = image[row_beg:row_beg + vcrop_size[0]]
            row_beg = (height - vcrop_size[0])
            vcrop_bottom = image[row_beg:]
            hcrop_range = [(width - hcrop_size[1]) // 2, (width - hcrop_size[1]) // 2 + hcrop_size[1]]
            hcrop_image = image[:, hcrop_range[0]:hcrop_range[1]]

            cv2.imshow("original", image)
            cv2.imshow("vertical middle", vcrop_middle)
            cv2.imshow("vertical bottom", vcrop_bottom)
            cv2.imshow("horizontal crop", hcrop_image)
            key = cv2.waitKey(2000)
            if key == ord('q'):
                break


if __name__ == "__main__":
    test_kitti_raw_reader()
    # test_kitti_odom_reader()
    # test_kitti_raw_synthesis()
    # test_crop_image()


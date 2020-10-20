import os.path as op
import numpy as np
from scipy import sparse
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from tfrecords.readers.reader_base import DataReaderBase
from utils.util_class import MyExceptionToCatch

T_C2V = tf.constant([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
FRONT_IND = 0


class WaymoReader(DataReaderBase):
    def __init__(self, split=""):
        super().__init__(split)
        self.tfr_dataset = None
        self.frame_buffer = dict()
        self.latest_index = -1

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """314314
        prepare variables to read a new sequence data
        """
        self.tfr_dataset = self._get_dataset(drive_path)
        self.tfr_dataset = iter(self.tfr_dataset)
        self.latest_index = -1
        # self.frame_names =
        # self.intrinsic =
        # self.T_left_right =

    def num_frames_(self):
        return 50000

    def get_range_(self):
        return range(2, self.num_frames_()-2)

    def get_image(self, index, right=False):
        if right: return None
        frame = self._get_frame(index)
        front_image = tf.image.decode_jpeg(frame.images[0].image)
        front_image = cv2.cvtColor(front_image.numpy(), cv2.COLOR_RGB2BGR)
        return front_image.astype(np.uint8)

    def get_pose(self, index, right=False):
        if right: return None
        frame = self._get_frame(index)
        pose_c2w = tf.reshape(frame.images[0].pose.transform, (4, 4)) @ T_C2V
        pose_c2w = pose_c2w.numpy()
        return pose_c2w.astype(np.float32)

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        if right: return None
        frame = self._get_frame(index)
        depth = get_waymo_depth_map(frame, srcshape_hw, dstshape_hw, intrinsic)
        depth = depth[..., np.newaxis]
        return depth.astype(np.float32)

    def get_intrinsic(self, index=0, right=False):
        if right: return None
        frame = self._get_frame(index)
        intrin = frame.context.camera_calibrations[0].intrinsic
        intrin = np.array([[intrin[0], 0, intrin[2]], [0, intrin[1], intrin[3]], [0, 0, 1]])
        return intrin.astype(np.float32)

    def get_stereo_extrinsic(self, index=0):
        return None

    def get_filename(self, example_index):
        return None

    """
    Private methods used inside this class
    """
    def _get_dataset(self, drive_path):
        filenames = tf.io.gfile.glob(f"{drive_path}/*.tfrecord")
        print("[WaymoReader._get_dataset] read tfrecords in", op.basename(drive_path), filenames)
        dataset = tf.data.TFRecordDataset(filenames, compression_type='')
        return dataset

    def _get_frame(self, index):
        if index in self.frame_buffer:
            frame = self.frame_buffer[index]
            time_of_day = f"{frame.context.stats.time_of_day}"
            if time_of_day != "Day":
                raise MyExceptionToCatch(f"time_of_day is not Day: {time_of_day}")
            return frame

        if (index == self.latest_index + 1) or self.latest_index < 0:
            # add a new frame
            frame_data = self.tfr_dataset.__next__()
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))
            self.frame_buffer[index] = frame
            if index - 20 in self.frame_buffer:
                self.frame_buffer.pop(index - 20, None)
            self.latest_index = index
            time_of_day = f"{frame.context.stats.time_of_day}"
            if time_of_day != "Day":
                raise MyExceptionToCatch(f"time_of_day is not Day: {time_of_day}")
            # remove an old frame
            return frame

        assert 0, f"frame index is not consecutive: {self.latest_index} to {index}"


def get_waymo_depth_map(frame, srcshape_hw, dstshape_hw, intrinsic):
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

    # extract LiDAR points projected to camera[FRONT_IND]
    camera_mask = np.equal(cp_points[:, 0], frame.images[FRONT_IND].name)
    points_veh = points_veh[camera_mask]

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


# ======================================================================
import cv2
import utils.util_funcs as uf
from config import opts


def test_waymo_reader():
    for di in range(0, 28):
        drive_path = f"/media/ian/IanBook2/datasets/waymo/training_{di:04d}"
        print("\n!!! New drive start !!!", drive_path)
        reader = WaymoReader("train")
        reader.init_drive(drive_path)
        pose_bef = np.zeros((4, 4))
        for fi in range(50000):
            try:
                frame = reader._get_frame(fi)
                image = reader.get_image(fi)
                pose = reader.get_pose(fi)
                intrinsic = reader.get_intrinsic(fi)
                depth = reader.get_depth(fi, image.shape[:2], opts.get_img_shape("HW", "waymo"), intrinsic)
            except StopIteration as si:
                print("StopIteration:", si)
                break

            dist = np.linalg.norm(pose[:3, 3] - pose_bef[:3, 3], axis=0)
            msg = f"[test_waymo_reader] drive: {di}, frame: {fi}, dist: {dist:1.3f}"
            msg += f", weather: {frame.context.stats.weather}, time: {frame.context.stats.time_of_day}"
            uf.print_progress_status(msg)
            view = image[0:-1:5, 0:-1:5, :]
            cv2.imshow("image", view)
            cv2.imshow("depth", depth)
            if dist < 5:
                key = cv2.waitKey(10)
            else:
                key = cv2.waitKey(2000)
            if key == ord('q'):
                break
            pose_bef = pose


if __name__ == "__main__":
    test_waymo_reader()


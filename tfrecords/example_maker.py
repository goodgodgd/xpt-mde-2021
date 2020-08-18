import numpy as np
import cv2

from tfrecords.readers.kitti_reader import KittiRawReader, KittiOdomReader
from tfrecords.readers.city_reader import CityscapesReader
from tfrecords.readers.waymo_reader import WaymoReader
from tfrecords.readers.driving_reader import DrivingStereoReader
from utils.convert_pose import pose_matr2rvec
from tfrecords.tfr_util import show_example


class ExampleMaker:
    def __init__(self, dataset, split, shwc_shape, data_keys, reader_args=None, crop=False):
        self.dataset = dataset
        self.split = split
        self.shwc_shape = shwc_shape
        self.data_keys = data_keys
        self.data_reader = WaymoReader()
        self.reader_args = reader_args
        self.crop = crop

    def init_reader(self, drive_path):
        self.data_reader = self.data_reader_factory()
        self.data_reader.init_drive(drive_path)

    def data_reader_factory(self):
        if self.dataset == "kitti_raw":
            return KittiRawReader(self.split, self.reader_args)     # srcpath
        elif self.dataset == "kitti_odom":
            return KittiOdomReader(self.split, self.reader_args)     # split and ZipFile object
        elif self.dataset.startswith("cityscapes"):
            return CityscapesReader(self.split, self.reader_args)   # split and ZipFile object
        elif self.dataset == "waymo":
            return WaymoReader(self.split)
        elif self.dataset == "driving_stereo":
            return DrivingStereoReader(self.split)
        else:
            assert 0, f"[data_reader_factory] invalid dataset name {self.dataset}"

    def num_frames(self):
        return self.data_reader.num_frames_()

    def get_range(self):
        return self.data_reader.get_range_()

    def get_example(self, index):
        frame_id, frame_seq_ids = self.make_snippet_ids(index)
        example = dict()
        example["image"], raw_shape_hwc = self.load_snippet_images(frame_seq_ids)
        example["intrinsic"] = self.load_intrinsic(frame_id, raw_shape_hwc)
        if "depth_gt" in self.data_keys:
            example["depth_gt"] = self.load_depth_map(frame_id, raw_shape_hwc)
        if "pose_gt" in self.data_keys:
            example["pose_gt"] = self.load_snippet_poses(frame_seq_ids)
        if "image_R" in self.data_keys:
            example["image_R"], _ = self.load_snippet_images(frame_seq_ids, right=True)
        if "intrinsic_R" in self.data_keys:
            example["intrinsic_R"] = self.load_intrinsic(frame_id, raw_shape_hwc, right=True)
        if "pose_gt_R" in self.data_keys:
            example["pose_gt_R"] = self.load_snippet_poses(frame_seq_ids, right=True)
        if "stereo_T_LR" in self.data_keys:
            example["stereo_T_LR"] = self.data_reader.get_stereo_extrinsic(frame_id)

        if index % 100 == 10:
            show_example(example, 200)
        if index % 500 == 10:
            print("\nintrinsic:\n", example["intrinsic"])
            if example["pose_gt"] is not None:
                print("pose\n", pose_matr2rvec(example["pose_gt"]))

        example = self.verify_snippet(example)
        return example

    def make_snippet_ids(self, frame_index):
        frame_id = self.data_reader.index_to_id(frame_index)
        halflen = self.shwc_shape[0] // 2
        max_frame_index = self.num_frames() - 1
        frame_seq_ids = np.arange(frame_id-halflen, frame_id+halflen+1)
        frame_seq_ids = np.clip(frame_seq_ids, 0, max_frame_index).tolist()
        return frame_id, frame_seq_ids

    def load_snippet_images(self, frame_ids, right=False):
        image_seq = []
        raw_shape = self.shwc_shape[1:]
        dstsize_wh = (self.shwc_shape[2], self.shwc_shape[1])

        for fid in frame_ids:
            image = self.data_reader.get_image(fid, right=right)
            if image is None:
                return None
            raw_shape = image.shape
            if self.crop:
                yxhw = self.crop_yxhw_range(image.shape)
                image = image[yxhw[0]:yxhw[0] + yxhw[2], yxhw[1]:yxhw[1] + yxhw[3]]
            image = cv2.resize(image, dstsize_wh)
            image_seq.append(image)
        # move target image to the bottom
        target_index = self.shwc_shape[0] // 2
        target_image = image_seq.pop(target_index)
        image_seq.append(target_image)
        image_seq = np.concatenate(image_seq, axis=0).astype(np.uint8)
        return image_seq, raw_shape

    def load_intrinsic(self, index, raw_shape_hwc, right=False):
        intrinsic = self.data_reader.get_intrinsic(index, right=right)
        if intrinsic is None:
            return None

        dst_shape_hw = self.shwc_shape[1:3]
        src_shape_hw = raw_shape_hwc[:2]
        if self.crop:
            yxhw = self.crop_yxhw_range(raw_shape_hwc)
            if index == 10:
                print("\ncrop image yxhw:", yxhw)
            intrinsic[0, 2] = intrinsic[0, 2] - yxhw[1]  # cx
            intrinsic[1, 2] = intrinsic[1, 2] - yxhw[0]  # cy
            src_shape_hw = yxhw[2:]

        # scale fx, cx
        intrinsic[0] = intrinsic[0] * dst_shape_hw[1] / src_shape_hw[1]
        # scale fy, cy
        intrinsic[1] = intrinsic[1] * dst_shape_hw[0] / src_shape_hw[0]
        return intrinsic.astype(np.float32)

    def load_snippet_poses(self, frame_ids, right=False):
        pose_seq = []
        for fid in frame_ids:
            pose = self.data_reader.get_pose(fid, right=right)
            if pose is None:
                return None
            pose_seq.append(pose)
        target_index = self.shwc_shape[0] // 2
        target_pose = pose_seq.pop(target_index)
        # poses that transform a point from target to source frame
        pose_seq = [np.linalg.inv(pose) @ target_pose for pose in pose_seq]
        pose_seq = np.stack(pose_seq, axis=0)
        return pose_seq.astype(np.float32)

    def load_depth_map(self, index, raw_shape_hwc):
        intrinsic = self.data_reader.get_intrinsic(index)
        if intrinsic is None: return None
        depth_map = self.data_reader.get_depth(index, raw_shape_hwc[:2], self.shwc_shape[1:3], intrinsic)
        if depth_map is None: return None
        return depth_map.astype(np.float32)

    def verify_snippet(self, example):
        if self.dataset is "waymo":
            poses = example["pose_gt"]
            positions = poses[:, :3, 3]
            distances = np.linalg.norm(positions, axis=1)

            min_dist = np.min(distances)
            if min_dist < 0.2:
                return dict()   # empty dict means skip this frame

            max_dist = np.max(distances)
            if max_dist > 10.:
                print("\n  Change scene? distance=", max_dist)
                return dict()   # empty dict means skip this frame
        return example

    def crop_yxhw_range(self, raw_shape_hwc):
        raw_h, raw_w = raw_shape_hwc[:2]
        exm_h, exm_w = self.shwc_shape[1:3]
        # crop vertically: crop upper region of image
        # e.g. KITTI: (376, 1241) -> (310, 1241) -> (128, 512)
        if raw_w / raw_h < exm_w / exm_h:
            new_height = int(exm_h * raw_w / exm_w + 0.5)
            row_begin = int((raw_h - new_height) * 0.6)
            return row_begin, 0, new_height, raw_w
        # crop horizontally: crop both left and right sides
        # e.g. KITTI: (376, 1241) -> (376, 1128) -> (128, 384)
        else:
            new_width = int(exm_w * raw_h / exm_h + 0.5)
            col_begin = (raw_w - new_width) // 2
            return 0, col_begin, raw_h, new_width



# ======================================================================
import cv2
from config import opts
from utils.util_funcs import print_progress_status


# This test is FAILED !!!
def test_static_frames():
    data_keys = ["image", "intrinsic", "depth_gt", "image_R", "intrinsic_R", "stereo_T_LR", "decode_type"]
    shape_shwc = opts.get_img_shape("SHWC")
    maker = ExampleMaker("driving_stereo", "train", shape_shwc, data_keys)
    drive_path = "/media/ian/IanBook/datasets/raw_zips/driving_stereo/train-left-image/2018-07-16-15-18-53.zip"
    maker.init_reader(drive_path)
    frame_indices = maker.get_range()
    for index in frame_indices:
        try:
            example = maker.get_example(index)
            print_progress_status(f"index: {index} / {frame_indices[-1]}")
            check_static_snippet(example, shape_shwc)
        except ValueError as ve:
            print("\n[ValueError]", ve)


def check_static_snippet(example, shape_shwc):
    height = shape_shwc[1]
    image = example["image"]
    frame_bef = cv2.cvtColor(image[:height], cv2.COLOR_BGRA2GRAY)
    frame_cur = cv2.cvtColor(image[-height:], cv2.COLOR_BGRA2GRAY)
    frame_aft = cv2.cvtColor(image[-2*height:-height], cv2.COLOR_BGRA2GRAY)
    flow1 = cv2.calcOpticalFlowFarneback(frame_bef, frame_cur,
                                         flow=None, pyr_scale=0.5, levels=3, winsize=10,
                                         iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    flow2 = cv2.calcOpticalFlowFarneback(frame_cur, frame_aft,
                                         flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                         iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
    flow1_dist = np.sqrt(flow1[:, :, 0] * flow1[:, :, 0] + flow1[:, :, 1] * flow1[:, :, 1])
    flow2_dist = np.sqrt(flow2[:, :, 0] * flow2[:, :, 0] + flow2[:, :, 1] * flow2[:, :, 1])
    img_size = flow1.shape[0] * flow1.shape[1]
    valid1 = np.count_nonzero((2 < flow1_dist) & (flow1_dist < 50)) / img_size
    valid2 = np.count_nonzero((2 < flow2_dist) & (flow2_dist < 50)) / img_size
    print(f"valid flow ratio: {valid1:.4f}, {valid2:.4f}")
    if (valid1 < 0.5) or (valid1 < 0.5):
        print("!!! frame jumped !!!")
    cv2.imshow("snippet", example["image"])
    frame = np.concatenate([frame_bef, frame_cur, frame_aft], axis=0)
    cv2.imshow("frame bef cur aft", frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_static_frames()

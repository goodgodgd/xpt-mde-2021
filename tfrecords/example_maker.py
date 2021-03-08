import numpy as np
import cv2
from timeit import default_timer as timer

from tfrecords.readers.kitti_reader import KittiRawReader, KittiOdomReader
from tfrecords.readers.city_reader import CityscapesReader
from tfrecords.readers.waymo_reader import WaymoReader
from tfrecords.readers.driving_reader import DrivingStereoReader
from tfrecords.readers.a2d2_reader import A2D2Reader
from tfrecords.tfr_util import show_example, point_cloud_to_depth_map
from utils.util_class import MyExceptionToCatch


class ExampleMaker:
    def __init__(self, dataset, split, shwc_shape, data_keys, reader_args=None):
        self.dataset = dataset
        self.split = split
        self.shwc_shape = shwc_shape
        self.data_keys = data_keys
        self.data_reader = WaymoReader()
        self.reader_args = reader_args
        self.max_frame_id = 0

    def init_reader(self, drive_path):
        self.data_reader = self.data_reader_factory()
        self.data_reader.init_drive(drive_path)
        if len(self.get_range()) > 0:
            self.max_frame_id = self.get_range()[-1]

    def data_reader_factory(self):
        if self.dataset == "kitti_raw":
            return KittiRawReader(self.split, self.reader_args)     # srcpath
        elif self.dataset == "kitti_odom":
            return KittiOdomReader(self.split, self.reader_args)     # split and ZipFile object
        elif self.dataset.startswith("cityscapes"):
            return CityscapesReader(self.split, self.reader_args)   # split and ZipFile object
        elif self.dataset == "waymo":
            return WaymoReader(self.split)
        elif self.dataset == "a2d2":
            return A2D2Reader(self.split, self.reader_args)
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
        example["image"], rawshape_hw, rszshape_hw = self.load_snippet_images(frame_seq_ids)
        self.check_static_sequence(example)

        example["intrinsic"] = self.load_intrinsic(frame_id, rawshape_hw, rszshape_hw)
        if "depth_gt" in self.data_keys:
            example["depth_gt"] = self.load_depth_map(frame_id, rawshape_hw, rszshape_hw)
        if "pose_gt" in self.data_keys:
            example["pose_gt"] = self.load_snippet_poses(frame_seq_ids)
        if "image_R" in self.data_keys:
            example["image_R"], _, _ = self.load_snippet_images(frame_seq_ids, right=True)
        if "intrinsic_R" in self.data_keys:
            example["intrinsic_R"] = self.load_intrinsic(frame_id, rawshape_hw, rszshape_hw, right=True)
        if "depth_gt_R" in self.data_keys:
            example["depth_gt_R"] = self.load_depth_map(frame_id, rawshape_hw, rszshape_hw, right=True)
        if "pose_gt_R" in self.data_keys:
            example["pose_gt_R"] = self.load_snippet_poses(frame_seq_ids, right=True)
        if "stereo_T_LR" in self.data_keys:
            example["stereo_T_LR"] = self.data_reader.get_stereo_extrinsic(frame_id)

        # if index % 500 == 10:
        #     show_example(example, 0, print_param=True, max_height=0)
        # elif index % 100 == 10:
        #     show_example(example, 0)

        example = self.crop_example(example, rszshape_hw)

        if index % 500 == 10:
            show_example(example, 200, print_param=True, max_height=0, suffix="_crop")
        elif index % 100 == 10:
            show_example(example, 200, max_height=0, suffix="_crop")
        example = self.verify_snippet(example)
        return example

    def make_snippet_ids(self, frame_index):
        frame_id = self.data_reader.index_to_id(frame_index)
        halflen = self.shwc_shape[0] // 2
        # max_frame_id = list(self.get_range())[-1]
        if (self.dataset == "a2d2") or (self.dataset.startswith("cityscapes")):
            frame_seq_ids = np.arange(frame_id-halflen*2, frame_id+halflen*2+1, 2)
        else:
            frame_seq_ids = np.arange(frame_id - halflen, frame_id + halflen + 1)
        frame_seq_ids = np.clip(frame_seq_ids, 0, self.max_frame_id).tolist()
        return frame_id, frame_seq_ids

    def load_snippet_images(self, frame_ids, right=False):
        image_seq = []
        rawshape_hw, rszshape_hw = (), ()
        dstshape_hw = (self.shwc_shape[1], self.shwc_shape[2])

        for fid in frame_ids:
            image = self.data_reader.get_image(fid, right=right)
            if image is None:
                return None, 0, 0
            rawshape_hw = image.shape[:2]
            rszshape_hw = self.get_resize_shape(rawshape_hw, dstshape_hw)
            image = cv2.resize(image, (rszshape_hw[1], rszshape_hw[0]))
            image_seq.append(image)
        # move target image to the bottom
        target_index = self.shwc_shape[0] // 2
        target_image = image_seq.pop(target_index)
        image_seq.append(target_image)
        image_seq = np.concatenate(image_seq, axis=0).astype(np.uint8)
        return image_seq, rawshape_hw, rszshape_hw

    def get_resize_shape(self, rawshape_hw, dstshape_hw):
        raw_ratio = rawshape_hw[1] / rawshape_hw[0]
        dst_ratio = dstshape_hw[1] / dstshape_hw[0]
        if np.abs(dst_ratio - raw_ratio) < 0.05:
            return dstshape_hw
        elif dst_ratio > raw_ratio:     # if dst is wider
            return int(rawshape_hw[0] * dstshape_hw[1] / rawshape_hw[1] + 0.5), dstshape_hw[1]
        else:                           # if dst is taller
            return dstshape_hw[0], int(rawshape_hw[1] * dstshape_hw[0] / rawshape_hw[0] + 0.5)

    def check_static_sequence(self, example):
        image_seq = example["image"]
        snippet, height, width, _ = self.shwc_shape
        height = image_seq.shape[0] // snippet
        num_src = snippet - 1
        dynamic_frames = 0
        target_frame = image_seq[(num_src * height):]
        y_border = height // 3
        diff_thresh = height * width // 50

        # create blurred diff images
        target_smooth = cv2.GaussianBlur(cv2.GaussianBlur(target_frame, (3, 3), 0), (3, 3), 0).astype(np.int32)
        for i in range(snippet):
            src_frame = image_seq[(i * height):(i * height + height)]
            src_smooth = cv2.GaussianBlur(cv2.GaussianBlur(src_frame, (3, 3), 0), (3, 3), 0).astype(np.int32)
            imdiff = np.absolute(target_smooth - src_smooth)
            diffmap = np.sum(imdiff[:y_border], axis=2)
            diff_pixels = np.sum(diffmap > 20).astype(int)
            if diff_pixels > diff_thresh:
                dynamic_frames += 1
        if dynamic_frames < 2:
            raise MyExceptionToCatch("[check_static_sequence] static sequence")

    def load_intrinsic(self, index, rawshape_hw, rszshape_hw, right=False):
        intrinsic_raw = self.data_reader.get_intrinsic(index, right=right)
        if intrinsic_raw is None:
            return None
        intrinsic = intrinsic_raw.copy()
        # scale fx, cx
        intrinsic[0] = intrinsic[0] * rszshape_hw[1] / rawshape_hw[1]
        # scale fy, cy
        intrinsic[1] = intrinsic[1] * rszshape_hw[0] / rawshape_hw[0]
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

    def load_depth_map(self, index, rawshape_hw, rszshape_hw, right=False):
        intrinsic = self.data_reader.get_intrinsic(index, right)
        if intrinsic is None: return None
        intrinsic_rsz = self.rescale_intrinsic(intrinsic, rawshape_hw, rszshape_hw)
        point_cloud = self.data_reader.get_point_cloud(index, right)
        if point_cloud is None: return None
        depth_map = point_cloud_to_depth_map(point_cloud, intrinsic_rsz, rszshape_hw)
        # depth_map = self.data_reader.get_depth(index, rawshape_hw, rszshape_hw, intrinsic, right)
        if depth_map.ndim == 2:
            depth_map = depth_map[..., np.newaxis]
        return depth_map.astype(np.float32)

    def rescale_intrinsic(self, intrinsic, rawshape_hw, rszshape_hw):
        intrinsic_rsz = intrinsic.copy()
        # rescale fx, cx
        intrinsic_rsz[0] *= (rszshape_hw[1] / rawshape_hw[1])
        # rescale fy, cy
        intrinsic_rsz[1] *= (rszshape_hw[0] / rawshape_hw[0])
        return intrinsic_rsz

    def verify_snippet(self, example):
        if self.dataset is "waymo":
            poses = example["pose_gt"]
            positions = poses[:, :3, 3]
            distances = np.linalg.norm(positions, axis=1)

            min_dist = np.min(distances)
            if min_dist < 0.2:
                raise MyExceptionToCatch("[verify_snippet] poses is not moving")

            max_dist = np.max(distances)
            if max_dist > 10.:
                print("\n  Change scene? distance=", max_dist)
                raise MyExceptionToCatch("[verify_snippet] scene is changing")

        example = {key: val for key, val in example.items() if val is not None}
        return example

    def crop_example(self, example, rszshape_hw):
        if rszshape_hw == self.shwc_shape[1:3]:
            return example

        cy, cx, ch, cw = self.get_crop_range(rszshape_hw)

        def crop_image(image):
            image5d = image.reshape(-1, rszshape_hw[0], rszshape_hw[1], 3)
            image_crop = image5d[:, cy:cy + ch, cx:cx + cw]
            # print("crop_image:", image.shape, image5d.shape, image_crop.shape, cy, cx, ch, cw)
            image_crop = image_crop.reshape(-1, cw, 3)
            return image_crop

        example["image"] = crop_image(example["image"])
        if "image_R" in example and example["image_R"] is not None:
            example["image_R"] = crop_image(example["image_R"])

        def crop_intrinsic(intrinsic_):
            intrinsic = np.copy(intrinsic_)
            intrinsic[0, 2] = intrinsic[0, 2] - cx
            intrinsic[1, 2] = intrinsic[1, 2] - cy
            return intrinsic

        example["intrinsic"] = crop_intrinsic(example["intrinsic"])
        if "intrinsic_R" in example and example["intrinsic_R"] is not None:
            example["intrinsic_R"] = crop_intrinsic(example["intrinsic_R"])

        if "depth_gt" in example and example["depth_gt"] is not None:
            example["depth_gt"] = example["depth_gt"][cy:cy + ch, cx:cx + cw]
        if "depth_gt_R" in example and example["depth_gt_R"] is not None:
            example["depth_gt_R"] = example["depth_gt_R"][cy:cy + ch, cx:cx + cw]

        return example

    def get_crop_range(self, rszshape_hw):
        rsz_h, rsz_w = rszshape_hw
        dst_h, dst_w = self.shwc_shape[1:3]

        if self.dataset.startswith("kitti"):
            # crop vertically
            if (rsz_h > dst_h) and (rsz_w == dst_w):
                row_beg = int((rsz_h - dst_h) * 0.7)    # remove sky area in image top
                return row_beg, 0, dst_h, dst_w
            # crop horizontally: crop both left and right sides
            else:
                col_beg = (rsz_w - dst_w) // 2
                return 0, col_beg, dst_h, dst_w
        elif (self.dataset == "a2d2") or (self.dataset.startswith("cityscapes")):
            # crop vertically
            if (rsz_h > dst_h) and (rsz_w == dst_w):    # remove vehicle part in image bottom
                return 0, 0, dst_h, dst_w
            # crop horizontally: crop both left and right sides
            else:
                col_beg = (rsz_w - dst_w) // 2
                return 0, col_beg, dst_h, dst_w

        elif self.dataset == "driving_stereo":
            if (rsz_h > dst_h) and (rsz_w == dst_w):
                row_beg = 0
                return row_beg, 0, dst_h, dst_w
            # crop horizontally: crop both left and right sides
            else:
                col_beg = (rsz_w - dst_w) // 2
                return 0, col_beg, dst_h, dst_w

        else:
            assert 0, f"Wrong dataset to crop: {self.dataset}"


# ======================================================================
from config import opts
from utils.util_funcs import print_progress_status
import os.path as op


# This test is FAILED !!!
# def test_static_frames():
#     # TODO : Test by changing variables below
#     # delete_static_sequence(image_seq, pixel_threshold, static_img_count)
#     count = 0
#     drive_ids = ["0001", "0002", "0005", "0009"]
#     # data_keys = ["image", "intrinsic", "depth_gt", "image_R", "intrinsic_R", "stereo_T_LR", "decode_type"]
#     data_keys = ["image", "intrinsic", "depth_gt", "image_R", "intrinsic_R", "stereo_T_LR", "decode_type", "pose_gt"]
#     # change when testing : img shapes are different per datset. Try testing with different dataset
#     # kitti :
#     # shape_shwc = opts.get_img_shape("SHWC", "kitti_raw")
#     # maker = ExampleMaker("kitti_raw", "train", shape_shwc, data_keys, "/media/ian/IanBook2/datasets/kitti_raw_data")
#     # drive_path = "/media/ian/IanBook2/datasets/kitti_raw_data/2011_09_26.zip"
#     # maker.init_reader(("2011_09_26", "0087"))
#
#     # waymo
#     # shape_shwc = opts.get_img_shape()
#     shape_shwc = opts.get_img_shape("SHWC", "waymo")
#     #print(shape_shwc)
#     maker = ExampleMaker("waymo", "train", shape_shwc, data_keys)
#     drive_path = "/media/ian/IanBook2/datasets/waymo/training_0001"
#     maker.init_reader(drive_path)
#     frame_indices = maker.get_range()
#     print("len frame indices : ", len(frame_indices))
#     for index in frame_indices:
#         try:
#             example = maker.get_example(index)
#             print_progress_status(f"index: {index} / {frame_indices[-1]}")
#             # threshold ratio set
#             print(f"{index}=th example keys : ", example.keys())
#             if "image" not in list(example.keys()):
#                 pass
#             else:
#                 # delete_static_sequence(example["image"], 10, count)
#             # check_static_snippet(example, shape_shwc)
#             # print(example)
#         except ValueError as ve:
#             print("\n[ValueError]", ve)
#


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


# if __name__ == "__main__":
    # test_static_frames()

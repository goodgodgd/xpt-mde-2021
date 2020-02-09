import cv2
import numpy as np

import settings
from config import opts, get_raw_data_path
import prepare_data.kitti_util as ku
import utils.convert_pose as cp
from utils.util_class import WrongInputException

"""
KittiDataLoader: generates training example, main function is snippet_generator()
KittiReader: reads data from files
"""


def kitti_loader_factory(base_path, dataset, split):
    stereo = opts.STEREO
    if dataset == "kitti_raw" and split == "train":
        data_reader = ku.KittiRawTrainUtil(stereo)
    elif dataset == "kitti_raw" and split == "test":
        data_reader = ku.KittiRawTestUtil(stereo)
    elif dataset == "kitti_odom" and split == "train":
        data_reader = ku.KittiOdomTrainReader(stereo)
    elif dataset == "kitti_odom" and split == "test":
        data_reader = ku.KittiOdomTestReader(stereo)
    else:
        raise WrongInputException(f"Wrong dataset and split: {dataset}, {split}")

    if stereo:
        return KittiDataLoaderStereo(base_path, split, data_reader)
    else:
        return KittiDataLoader(base_path, split, data_reader)


class KittiDataLoader:
    def __init__(self, base_path, split, reader):
        self.base_path = base_path
        self.kitti_reader = reader
        self.drive_list = self.kitti_reader.list_drives(split, base_path)
        self.drive_path = ""
        self.frame_inds = []

    def load_drive(self, drive, snippet_len):
        self.kitti_reader.create_drive_loader(self.base_path, drive)
        self.drive_path = self.kitti_reader.make_drive_path(self.base_path, drive)
        self.frame_inds = self.kitti_reader.find_frame_indices(self.drive_path, snippet_len)
        if self.frame_inds.size > 1:
            print(f"frame_indices: {self.frame_inds[0]} ~ {self.frame_inds[-1]}")
        return self.frame_inds

    def example_generator(self, index, snippet_len):
        example = dict()
        example["index"] = index
        example["image"], raw_img_shape = self.load_snippet_frames(index, snippet_len)
        example["intrinsic"] = self.load_intrinsic(raw_img_shape)
        if self.kitti_reader.pose_avail:
            example["pose_gt"] = self.load_snippet_poses(index, snippet_len)
        if self.kitti_reader.depth_avail:
            example["depth_gt"] = self.load_frame_depth(index, self.drive_path, raw_img_shape)
        return example

    def load_snippet_frames(self, frame_idx, snippet_len):
        halflen = snippet_len//2
        frames = []
        raw_img_shape = ()
        for ind in range(frame_idx-halflen, frame_idx+halflen+1):
            frame = self.kitti_reader.get_image(ind)
            raw_img_shape = frame.shape[:2]
            frame = cv2.resize(frame, dsize=(opts.IM_WIDTH, opts.IM_HEIGHT), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)

        frames = np.concatenate(frames, axis=0)
        return frames, raw_img_shape

    def load_snippet_poses(self, frame_idx, snippet_len):
        halflen = snippet_len//2
        poses = []
        for ind in range(frame_idx-halflen, frame_idx+halflen+1):
            pose = self.kitti_reader.get_quat_pose(ind)
            poses.append(pose)

        poses = np.stack(poses, axis=0)
        # print("poses bef\n", poses)
        poses = self.to_local_pose(poses, halflen)
        # print("poses local\n", poses)
        return poses

    def to_local_pose(self, poses, target_index):
        tgt_to_src_poses = []
        target_pose_mat = cp.pose_quat2matr(poses[target_index])
        for pose in poses:
            cur_pose_mat = cp.pose_quat2matr(pose)
            tgt_to_src_mat = np.matmul(np.linalg.inv(cur_pose_mat), target_pose_mat)
            tgt_to_src_qpose = cp.pose_matr2quat(tgt_to_src_mat)
            tgt_to_src_poses.append(tgt_to_src_qpose)

        tgt_to_src_poses = np.stack(tgt_to_src_poses, axis=0)
        return tgt_to_src_poses

    def load_frame_depth(self, frame_idx, drive_path, raw_img_shape):
        dst_shape = (opts.IM_HEIGHT, opts.IM_WIDTH)
        depth_map = self.kitti_reader.get_depth_map(frame_idx, drive_path, raw_img_shape, dst_shape)
        return depth_map

    def load_intrinsic(self, raw_img_shape):
        intrinsic = self.kitti_reader.get_intrinsic()
        sx = opts.IM_WIDTH / raw_img_shape[1]
        sy = opts.IM_HEIGHT / raw_img_shape[0]
        out = intrinsic.copy()
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out


class KittiDataLoaderStereo(KittiDataLoader):
    def __init__(self, base_path, split, reader):
        super().__init__(base_path, split, reader)

    def load_drive(self, drive, snippet_len):
        self.kitti_reader.create_drive_loader(self.base_path, drive)
        self.drive_path = self.kitti_reader.make_drive_path(self.base_path, drive)
        self.frame_inds = self.kitti_reader.find_frame_indices(self.drive_path, snippet_len)
        if self.frame_inds.size > 1:
            print(f"frame_indices: {self.frame_inds[0]} ~ {self.frame_inds[-1]}")
        return self.frame_inds

    def example_generator(self, index, snippet_len):
        example = dict()
        example["index"] = index
        example["image"], raw_img_shape = self.load_snippet_frames_stereo(index, snippet_len)
        example["intrinsic"] = self.load_intrinsic_stereo(raw_img_shape)
        if self.kitti_reader.pose_avail:
            example["pose_gt"] = self.load_snippet_poses_stereo(index, snippet_len)
        if self.kitti_reader.depth_avail:
            example["depth_gt"] = self.load_frame_depth_stereo(index, self.drive_path, raw_img_shape)
        return example

    def load_snippet_frames_stereo(self, frame_idx, snippet_len):
        halflen = snippet_len//2
        frames = []
        raw_img_shape = ()
        for ind in range(frame_idx-halflen, frame_idx+halflen+1):
            img_lef, img_rig = self.kitti_reader.get_image(ind)
            raw_img_shape = img_lef.shape[:2]
            img_lef = cv2.resize(img_lef, (opts.IM_WIDTH, opts.IM_HEIGHT), interpolation=cv2.INTER_LINEAR)
            img_lef = cv2.cvtColor(img_lef, cv2.COLOR_RGB2BGR)
            img_rig = cv2.resize(img_rig, (opts.IM_WIDTH, opts.IM_HEIGHT), interpolation=cv2.INTER_LINEAR)
            img_rig = cv2.cvtColor(img_rig, cv2.COLOR_RGB2BGR)
            frame = np.concatenate([img_lef, img_rig], axis=1)
            frames.append(frame)

        frames = np.concatenate(frames, axis=0)
        return frames, raw_img_shape

    def load_intrinsic_stereo(self, raw_img_shape):
        intrin_lef, intrin_rig = self.kitti_reader.get_intrinsic()
        intrinsic = np.concatenate([intrin_lef, intrin_rig], axis=1)
        sx = opts.IM_WIDTH / raw_img_shape[1]
        sy = opts.IM_HEIGHT / raw_img_shape[0]
        intrinsic[0, :] = intrinsic[0, :] * sx
        intrinsic[1, :] = intrinsic[1, :] * sy
        return intrinsic

    def load_snippet_poses_stereo(self, frame_idx, snippet_len):
        halflen = snippet_len//2
        pose_seq_lef = []
        pose_seq_rig = []
        for ind in range(frame_idx-halflen, frame_idx+halflen+1):
            pose_lef, pose_rig = self.kitti_reader.get_quat_pose(ind)
            pose_seq_lef.append(pose_lef)
            pose_seq_rig.append(pose_rig)

        pose_seq_lef = np.stack(pose_seq_lef, axis=0)
        pose_seq_rig = np.stack(pose_seq_rig, axis=0)
        pose_seq_lef = self.to_local_pose(pose_seq_lef, halflen)
        pose_seq_rig = self.to_local_pose(pose_seq_rig, halflen)
        poses = np.concatenate([pose_seq_lef, pose_seq_rig], axis=1)
        return poses

    def load_frame_depth_stereo(self, frame_idx, drive_path, raw_img_shape):
        dst_shape = (opts.IM_HEIGHT, opts.IM_WIDTH)
        depth_lef, depth_rig = self.kitti_reader.get_depth_map(frame_idx, drive_path, raw_img_shape, dst_shape)
        depth = np.concatenate([depth_lef, depth_rig], axis=1)
        return depth


def test():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    dataset = "kitti_odom"
    loader = KittiDataLoader(get_raw_data_path(dataset), dataset, "train")

    for drive in loader.drive_list:
        frame_indices = loader.load_drive(drive, opts.SNIPPET_LEN)
        if frame_indices.size == 0:
            print("this drive is EMPTY")
            continue

        for index, i in enumerate(frame_indices):
            snippet = loader.snippet_generator(index, opts.SNIPPET_LEN)
            index = snippet["index"]
            frames = snippet["image"]
            poses = snippet["pose_gt"]
            depths = snippet["depth_gt"]
            intrinsic = snippet["intrinsic"]
            print(f"frame {index}: concatenated image shape={frames.shape}, pose shape={poses.shape}")
            print("pose", poses[:3, :])
            print("intrinsic", intrinsic)
            cv2.imshow("frame", frames)
            key = cv2.waitKey(1000)
            if key == ord('q'):
                return
            if key == ord('s'):
                break


if __name__ == "__main__":
    test()

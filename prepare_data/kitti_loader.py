import cv2
import numpy as np

import settings
from config import opts, get_raw_data_path
import prepare_data.kitti_util as ku
import utils.convert_pose as cp


class KittiDataLoader:
    def __init__(self, base_path, dataset, split):
        self.base_path = base_path
        self.kitti_reader = self.kitti_reader_factory(dataset, split)
        self.drive_list = self.kitti_reader.list_drives(split, base_path)
        self.drive_path = ""
        self.frame_inds = []

    def kitti_reader_factory(self, dataset, split):
        if dataset == "kitti_raw" and split == "train":
            return ku.KittiRawTrainUtil()
        elif dataset == "kitti_raw" and split == "test":
            return ku.KittiRawTestUtil()
        elif dataset == "kitti_odom" and split == "train":
            return ku.KittiOdomTrainUtil()
        elif dataset == "kitti_odom" and split == "test":
            return ku.KittiOdomTestUtil()
        else:
            raise ValueError()

    def load_drive(self, drive, snippet_len):
        self.kitti_reader.create_drive_loader(self.base_path, drive)
        self.drive_path = self.kitti_reader.get_drive_path(self.base_path, drive)
        self.frame_inds = self.kitti_reader.frame_indices(self.drive_path, snippet_len)
        if self.frame_inds.size > 1:
            print(f"frame_indices: {self.frame_inds[0]} ~ {self.frame_inds[-1]}")
        return self.frame_inds

    def snippet_generator(self, index, snippet_len):
        example = dict()
        example["index"] = index
        example["frames"], raw_img_shape = self.load_snippet_frames(index, snippet_len)
        example["intrinsic"] = self.load_intrinsic(raw_img_shape)
        if self.kitti_reader.pose_avail:
            example["gt_poses"] = self.load_snippet_poses(index, snippet_len)
        if self.kitti_reader.depth_avail:
            example["gt_depth"] = self.load_frame_depth(index, self.drive_path, raw_img_shape)
        return example

    def load_snippet_frames(self, frame_idx, snippet_len):
        halflen = snippet_len//2
        frames = []
        raw_img_shape = ()
        for ind in range(frame_idx-halflen, frame_idx+halflen+1):
            frame = self.kitti_reader.get_image(ind)
            frame = np.array(frame[0])
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
        #########
        depth_map = self.kitti_reader.load_depth_map(frame_idx, drive_path, raw_img_shape, dst_shape)
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
            frames = snippet["frames"]
            poses = snippet["gt_poses"]
            depths = snippet["gt_depth"]
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

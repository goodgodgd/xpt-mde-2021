import os.path as op
import cv2
import numpy as np

import settings
from config import opts
import prepare_data.kitti_depth_generator as kdg
import prepare_data.kitti_util as ku


class KittiDataLoader:
    def __init__(self, base_path, dataset, split):
        self.base_path = base_path
        self.kitti_util = self.kitti_util_factory(dataset, split)
        self.drive_list = self.kitti_util.list_drives(split)
        self.drive_loader = None
        self.drive_path = ""

    def kitti_util_factory(self, dataset, split):
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

    def load_drive(self, drive):
        self.drive_loader = self.kitti_util.create_drive_loader(self.base_path, drive)
        self.drive_path = self.kitti_util.get_drive_path(self.base_path, drive)

    def snippet_generator(self, snippet_len):
        print("=" * 50)
        frame_inds = self.kitti_util.frame_indices(self.drive_path, snippet_len)
        for ind in frame_inds:
            print("=" * 10, ind)
            example = dict()
            example["frames"], raw_img_shape = self.load_snippet_frames(ind, snippet_len)
            example["gt_poses"] = self.load_snippet_poses(ind, snippet_len)
            example["gt_depth"] = self.load_frame_depth(ind, self.drive_path, raw_img_shape)
            example["intrinsic"] = self.load_intrinsic(raw_img_shape)
            yield example

    def load_snippet_frames(self, frindex, snippet_len):
        halflen = snippet_len//2
        frames = []
        raw_img_shape = ()
        for ind in range(frindex-halflen, frindex+halflen+1):
            frame = self.drive_loader.get_rgb(ind)
            frame = np.array(frame[0])
            raw_img_shape = frame.shape[:2]
            frame = cv2.resize(frame, dsize=(opts.IM_WIDTH, opts.IM_HEIGHT), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        frames = np.concatenate(frames, axis=0)
        return frames, raw_img_shape

    def load_snippet_poses(self, frindex, snippet_len):
        halflen = snippet_len//2
        poses = []
        for ind in range(frindex-halflen, frindex+halflen+1):
            pose = self.kitti_util.get_pose(self.drive_loader, ind)
            poses.append(pose)

        poses = np.stack(poses, axis=0)
        return poses

    def load_frame_depth(self, frindex, drive_path, raw_img_shape):
        # if dataset is kitti_odom, no depth available
        if drive_path.split("/")[-2] == "sequences":
            return None
        calib_dir = op.dirname(drive_path)
        velo_data = self.drive_loader.get_velo(frindex)
        depth_map = kdg.generate_depth_map(velo_data, calib_dir, raw_img_shape)
        print(f"depthmap shape={depth_map.shape}, mean={np.mean(depth_map, axis=None)}")
        return depth_map

    def load_intrinsic(self, raw_img_shape):
        intrinsic = self.drive_loader.calib.P_rect_20[:, :3]
        sx = opts.IM_WIDTH / raw_img_shape[1]
        sy = opts.IM_HEIGHT / raw_img_shape[0]
        out = intrinsic.copy()
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        print(f"scaled intrinsic: sx={sx:1.4f}, sy={sy:1.4f}\n", out)
        return 0


def test():
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    loader = KittiDataLoader(opts.RAW_DATASET_PATH, opts.DATASET, "train")

    for drive in loader.drive_list:
        print("drive:", drive)
        loader.load_drive(drive)
        for snippet in loader.snippet_generator(opts.SNIPPET_LEN):
            frames = snippet["frames"]
            poses = snippet["gt_poses"]
            depths = snippet["gt_depth"]
            intrinsic = snippet["intrinsic"]
            print(f"frame: concatenated image shape={frames.shape}, pose shape={poses.shape}")
            print(poses[:3, :])
            cv2.imshow("frame", frames)
            key = cv2.waitKey(1000)
            if key == ord('q'):
                return
            if key == ord('s'):
                break


if __name__ == "__main__":
    test()

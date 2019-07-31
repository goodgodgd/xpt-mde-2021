import os
import os.path as op
import cv2
import numpy as np

from config import opts
from kitti_loader import KittiDataLoader
from utils.util_funcs import print_progress

# TODO: odometry에 static frame 목록 만들기
'''
https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
def draw_flow(img, flow, step=16):
    ...

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
https://eehoeskrap.tistory.com/124
'''


def prepare_input_data():
    for dataset in ["kitti_raw", "kitti_odom"]:
        for split in ["train", "test"]:
            loader = KittiDataLoader(opts.get_dataset_path(dataset), dataset, split)
            prepare_and_save_snippets(loader, dataset, split)


def prepare_and_save_snippets(loader, dataset, split):
    dstpath = op.join(opts.RESULT_PATH, "srcdata", f"{dataset}_{split}")
    os.makedirs(dstpath, exist_ok=True)

    for drive in loader.drive_list:
        snippet_path, pose_path, depth_path = get_destination_paths(dstpath, dataset, drive)
        print("drive path:", snippet_path)
        if op.isdir(snippet_path):
            print("this drive has already prepared")
            continue

        frame_indices = loader.load_drive(drive, opts.SNIPPET_LEN)
        if frame_indices.size == 0:
            print("this drive is EMPTY")
            continue

        os.makedirs(snippet_path, exist_ok=True)
        os.makedirs(pose_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        print_progress(len(frame_indices), True)
        for index, i in enumerate(frame_indices):
            snippet = loader.snippet_generator(index, opts.SNIPPET_LEN)
            index = snippet["index"]
            frames = snippet["frames"]
            filename = op.join(snippet_path, f"{index:06d}.png")
            cv2.imwrite(filename, frames,)

            poses = snippet["gt_poses"]
            filename = op.join(pose_path, f"{index:06d}.txt")
            np.savetxt(filename, poses, fmt="%3.5f")

            depth = snippet["gt_depth"]
            mean_depth = 0
            if depth is not None:
                filename = op.join(depth_path, f"{index:06d}.npy")
                np.savetxt(filename, depth, fmt="%3.5f")
                mean_depth = np.mean(depth)

            intrinsic = snippet["intrinsic"]
            if not op.isfile(filename):
                print("##### intrinsic parameters\n", intrinsic)
                filename = op.join(snippet_path, "intrinsic.txt")
                np.savetxt(filename, intrinsic, fmt="%3.5f")

            cv2.imshow("snippet frames", frames)
            cv2.waitKey(1)
            print_progress(f"mean depth={mean_depth:0.3f}, {i}")


def get_destination_paths(dstpath, dataset, drive):
    if dataset == "kitti_raw":
        drive_path = op.join(dstpath, f"{drive[0]}_{drive[1]}")
    elif dataset == "kitti_odom":
        drive_path = op.join(dstpath, drive)
    else:
        raise ValueError()

    pose_path = op.join(drive_path, "pose")
    depth_path = op.join(drive_path, "depth")
    return drive_path, pose_path, depth_path


if __name__ == "__main__":
    prepare_input_data()

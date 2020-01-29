import os
import os.path as op
import cv2
import numpy as np
import shutil

import settings
from config import opts, get_raw_data_path
from prepare_data.kitti_loader import KittiDataLoader
from utils.util_funcs import print_progress_status


def prepare_kitti_data():
    for dataset in ["kitti_raw", "kitti_odom"]:
        for split in ["train", "test"]:
            loader = KittiDataLoader(get_raw_data_path(dataset), dataset, split)
            prepare_and_save_snippets(loader, dataset, split)
        create_validation_set(dataset)


def prepare_and_save_snippets(loader, dataset, split):
    dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_{split}")
    os.makedirs(dstpath, exist_ok=True)
    num_drives = len(loader.drive_list)

    for i, drive in enumerate(loader.drive_list):
        snippet_path, pose_path, depth_path = get_destination_paths(dstpath, dataset, drive)
        if op.isdir(snippet_path):
            print(f"this drive may have already prepared, check this path completed: {snippet_path}")
            continue

        print(f"\n{'=' * 50}\n[load drive] [{i}/{num_drives}] drive path: {snippet_path}")
        frame_indices = loader.load_drive(drive, opts.SNIPPET_LEN)
        if frame_indices.size == 0:
            print("this drive is EMPTY")
            continue

        os.makedirs(snippet_path, exist_ok=True)
        if loader.kitti_reader.pose_avail:
            os.makedirs(pose_path, exist_ok=True)
        if loader.kitti_reader.depth_avail:
            os.makedirs(depth_path, exist_ok=True)

        num_frames = len(frame_indices)
        for k, index in enumerate(frame_indices):
            snippet = loader.snippet_generator(index, opts.SNIPPET_LEN)
            index = snippet["index"]
            frames = snippet["frames"]
            filename = op.join(snippet_path, f"{index:06d}.png")
            cv2.imwrite(filename, frames)

            if "gt_poses" in snippet:
                poses = snippet["gt_poses"]
                filename = op.join(pose_path, f"{index:06d}.txt")
                np.savetxt(filename, poses, fmt="%3.5f")

            mean_depth = 0
            if "gt_depth" in snippet:
                depth = snippet["gt_depth"]
                filename = op.join(depth_path, f"{index:06d}.txt")
                np.savetxt(filename, depth, fmt="%3.5f")
                mean_depth = np.mean(depth)

            filename = op.join(snippet_path, "intrinsic.txt")
            if not op.isfile(filename):
                intrinsic = snippet["intrinsic"]
                print("##### intrinsic parameters\n", intrinsic)
                np.savetxt(filename, intrinsic, fmt="%3.5f")

            # cv2.imshow("snippet frames", frames)
            # cv2.waitKey(1)
            print_progress_status(f"- Progress: mean depth={mean_depth:0.3f}, {k}/{num_frames}")
        print("\t progress done")
    print(f"Data preparation of {dataset}_{split} is done")


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


def create_validation_set(dataset):
    srcpath = op.join(opts.DATAPATH_SRC, f"{dataset}_test")
    dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_val")

    if dataset == "kitti_raw":
        if os.path.exists(dstpath):
            os.unlink(dstpath)
        os.symlink(srcpath, dstpath)
    elif dataset == "kitti_odom":
        os.makedirs(dstpath, exist_ok=True)
        for drive in ["09", "10"]:
            shutil.copytree(os.path.join(srcpath, drive), os.path.join(dstpath, drive))

    print(f"\n### create validation split for {dataset}")


def prepare_single_dataset():
    dataset = "kitti_odom"
    split = "train"
    loader = KittiDataLoader(get_raw_data_path(dataset), dataset, split)
    prepare_and_save_snippets(loader, dataset, split)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    prepare_kitti_data()
    # prepare_single_dataset()

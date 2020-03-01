import os
import os.path as op
import cv2
import numpy as np
import shutil
import glob

import settings
from config import opts, get_raw_data_path
from prepare_data.kitti_loader import kitti_loader_factory
from utils.util_funcs import print_progress_status
from utils.util_class import PathManager


def prepare_kitti_data(dataset_in=None, split_in=None):
    datasets = ["kitti_raw", "kitti_odom"] if dataset_in is None else [dataset_in]
    splits = ["test", "train", "val"] if split_in is None else [split_in]
    for dataset in datasets:
        for split in splits:
            if split == "val":
                create_validation_set(dataset, "test")
            else:
                loader = kitti_loader_factory(get_raw_data_path(dataset), dataset, split)
                prepare_and_save_snippets(loader, dataset, split)


def prepare_and_save_snippets(loader, dataset, split):
    dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_{split}")
    os.makedirs(dstpath, exist_ok=True)
    num_drives = len(loader.drive_list)

    for i, drive in enumerate(loader.drive_list):
        pose_avail, depth_avail = loader.kitti_reader.pose_avail, loader.kitti_reader.depth_avail
        data_paths = get_destination_paths(dstpath, dataset, drive, pose_avail, depth_avail)
        snippet_path = data_paths[0]
        if op.isdir(snippet_path):
            print(f"this drive may have already prepared, check this path completed: {snippet_path}")
            continue

        print(f"\n{'=' * 50}\n[load drive] [{i+1}/{num_drives}] drive path: {snippet_path}")
        frame_indices = loader.load_drive(drive, opts.SNIPPET_LEN)
        num_frames = len(frame_indices)
        assert num_frames > 0

        with PathManager(data_paths) as pm:
            for k, index in enumerate(frame_indices):
                example = loader.example_generator(index, opts.SNIPPET_LEN)
                mean_depth = save_example(example, index, data_paths)
                print_progress_status(f"Progress: mean depth={mean_depth:0.3f}, index={index}, {k}/{num_frames}")
            # if set_ok() was NOT excuted, the generated path is removed
            pm.set_ok()
        print("")
    print(f"Data preparation of {dataset}_{split} is done")


def save_example(example, index, data_paths):
    snippet_path, pose_path, depth_path = data_paths
    frames = example["image"]
    filename = op.join(snippet_path, f"{index:06d}.png")
    cv2.imwrite(filename, frames)
    center_y, center_x = (opts.IM_HEIGHT // 2, opts.IM_WIDTH // 2)

    if "pose_gt" in example:
        poses = example["pose_gt"]
        filename = op.join(pose_path, f"{index:06d}.txt")
        np.savetxt(filename, poses, fmt="%3.5f")

    mean_depth = 0
    if "depth_gt" in example:
        depth = example["depth_gt"]
        filename = op.join(depth_path, f"{index:06d}.txt")
        np.savetxt(filename, depth, fmt="%3.5f")
        mean_depth = depth[center_y - 10:center_y + 10, center_x - 10:center_x + 10].mean()

    filename = op.join(snippet_path, "intrinsic.txt")
    if not op.isfile(filename):
        intrinsic = example["intrinsic"]
        print("intrinsic parameters\n", intrinsic)
        np.savetxt(filename, intrinsic, fmt="%3.5f")

    if "stereo_T_LR" in example:
        filename = op.join(snippet_path, "stereo_T_LR.txt")
        if not op.isfile(filename):
            extrinsic = example["stereo_T_LR"]
            np.savetxt(filename, extrinsic, fmt="%3.5f")

    # cv2.imshow("example frames", frames)
    # cv2.waitKey(1)
    return mean_depth


def get_destination_paths(dstpath, dataset, drive, pose_avail, depth_avail):
    if dataset == "kitti_raw":
        drive_path = op.join(dstpath, f"{drive[0]}_{drive[1]}")
    elif dataset == "kitti_odom":
        drive_path = op.join(dstpath, drive)
    else:
        raise ValueError()

    pose_path = op.join(drive_path, "pose") if pose_avail else None
    depth_path = op.join(drive_path, "depth") if depth_avail else None
    return drive_path, pose_path, depth_path


def create_validation_set(dataset, src_split):
    srcpath = op.join(opts.DATAPATH_SRC, f"{dataset}_{src_split}")
    dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_val")
    assert op.isdir(srcpath), f"[create_validation_set] src path does NOT exist {srcpath}"

    srcpattern = op.join(srcpath, "*", "*.png")
    srcfiles = glob.glob(srcpattern)
    # if files are too many, select frames
    if len(srcfiles) >= opts.VALIDATION_FRAMES:
        selinds = np.arange(0, opts.VALIDATION_FRAMES, 1) / opts.VALIDATION_FRAMES * len(srcfiles)
        selinds = selinds.astype(int)
        selected_files = [srcfiles[ind] for ind in selinds]
        srcfiles = selected_files

    num_files = len(srcfiles)
    print(f"num files: {num_files}, start creating validation set")
    for ind, srcimg in enumerate(srcfiles):
        imres = copy_file(srcimg, srcpath, dstpath)
        pores = copy_file(srcimg, srcpath, dstpath, "pose", "txt")
        deres = copy_file(srcimg, srcpath, dstpath, "depth", "txt")
        inres = copy_text(srcimg, srcpath, dstpath, "intrinsic.txt")
        stres = copy_text(srcimg, srcpath, dstpath, "stereo_T_LR.txt")
        print_progress_status(f"[create_validation_set] copy: {ind}/{num_files}, "
                              f"{srcimg.replace(opts.DATAPATH_SRC, '')}, "
                              f"{imres, pores, deres, inres, stres}")


def copy_file(imgfile, srcpath, dstpath, dirname=None, extension=None):
    srcfile = imgfile
    if dirname:
        srcfile = op.join(op.dirname(imgfile), dirname, op.basename(imgfile))
    if extension:
        srcfile = srcfile[:-3] + extension
    dstfile = srcfile.replace(srcpath, dstpath)
    if not op.isfile(srcfile):
        return 0

    if not op.isdir(op.dirname(dstfile)):
        os.makedirs(op.dirname(dstfile))
    shutil.copy(srcfile, dstfile)
    return 1


def copy_text(imgfile, srcpath, dstpath, filename):
    srcfile = op.join(op.dirname(imgfile), filename)
    dstfile = srcfile.replace(srcpath, dstpath)
    if not op.isfile(srcfile) or op.isfile(dstfile):
        return 0
    shutil.copy(srcfile, dstfile)
    return 1


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    prepare_kitti_data("kitti_odom", "val")

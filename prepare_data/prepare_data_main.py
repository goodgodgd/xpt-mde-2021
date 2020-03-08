import os
import os.path as op
import cv2
import numpy as np
import shutil
import glob

import settings
from config import opts, get_raw_data_path
from prepare_data.example_maker import dataset_reader_factory, example_maker_factory
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
                data_reader = dataset_reader_factory(get_raw_data_path(dataset), dataset, split)
                snippet_maker = example_maker_factory(get_raw_data_path(dataset))
                prepare_and_save_snippets(snippet_maker, data_reader, dataset, split)


def prepare_and_save_snippets(snippet_maker, data_reader, dataset, split):
    dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_{split}")
    os.makedirs(dstpath, exist_ok=True)
    drive_paths = data_reader.list_drive_paths()
    num_drives = len(drive_paths)
    half_len = opts.SNIPPET_LEN // 2

    for i, drive_path in enumerate(drive_paths):
        num_frames = data_reader.init_drive(drive_path)
        assert num_frames > 0

        data_paths = data_reader.make_saving_paths(dstpath, drive_path)
        image_path = data_paths[0]
        if op.isdir(image_path):
            print(f"this drive may have already prepared, check this path completed: {image_path}")
            continue

        print(f"\n{'=' * 50}\n[load drive] [{i+1}/{num_drives}] drive path: {image_path}")
        snippet_maker.set_reader(data_reader, num_frames)
        with PathManager(data_paths) as pm:
            frame_indices = range(half_len, num_frames - half_len) if split in "train" else range(0, num_frames)
            for index in frame_indices:
                example = snippet_maker.get_example(index)
                filename = data_reader.get_filename(index)
                mean_depth = save_example(example, filename, data_paths)
                print_progress_status(f"Progress: mean depth={mean_depth:0.3f}, file={filename} {index}/{num_frames}")

                if dataset == "cityscapes" and index > 30:
                    break

            # if set_ok() was NOT excuted, the generated path is removed
            pm.set_ok()
        print("")
    print(f"Data preparation of {dataset}_{split} is done")


def save_example(example, filename, data_paths):
    image_path, pose_path, depth_path = data_paths
    frames = example["image"]
    filepath = op.join(image_path, f"{filename}.png")
    cv2.imwrite(filepath, frames)
    center_y, center_x = (opts.IM_HEIGHT // 2, opts.IM_WIDTH // 2)

    if "pose_gt" in example:
        poses = example["pose_gt"]
        filepath = op.join(pose_path, f"{filename}.txt")
        np.savetxt(filepath, poses, fmt="%3.5f")

    mean_depth = 0
    if "depth_gt" in example:
        depth = example["depth_gt"]
        filepath = op.join(depth_path, f"{filename}.txt")
        np.savetxt(filepath, depth, fmt="%3.5f")
        mean_depth = depth[center_y - 10:center_y + 10, center_x - 10:center_x + 10].mean()

    filepath = op.join(image_path, "intrinsic.txt")
    if not op.isfile(filepath):
        intrinsic = example["intrinsic"]
        print("intrinsic parameters\n", intrinsic)
        np.savetxt(filepath, intrinsic, fmt="%3.5f")

    if "stereo_T_LR" in example:
        filepath = op.join(image_path, "stereo_T_LR.txt")
        if not op.isfile(filepath):
            extrinsic = example["stereo_T_LR"]
            np.savetxt(filepath, extrinsic, fmt="%3.5f")

    # cv2.imshow("example frames", frames)
    # cv2.waitKey(1)
    return mean_depth


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
    print("")
    print(f"Data preparation of {dataset}_val is done")


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
    prepare_kitti_data("cityscapes", "train_extra")

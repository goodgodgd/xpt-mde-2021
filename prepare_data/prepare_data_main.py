import os
import os.path as op
import cv2
import numpy as np
import shutil
import glob

import settings
from config import opts
from utils.util_funcs import print_progress_status
from utils.util_class import PathManager
import prepare_data.data_factories as df


def prepare_datasets():
    datasets = opts.DATASETS_TO_PREPARE
    for dataset, splits in datasets.items():
        for split in splits:
            if split == "val":
                create_validation_set(dataset, "test")
            else:
                dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_{split}")
                drive_lister = df.drive_lister_factory(dataset, split, opts.get_raw_data_path(dataset), dstpath)
                snippet_maker = df.example_maker_factory(opts.get_raw_data_path(dataset),
                                                         drive_lister.pose_avail, drive_lister.depth_avail)
                prepare_and_save_snippets(snippet_maker, drive_lister, dataset, split)


def prepare_and_save_snippets(snippet_maker, drive_lister, dataset, split):
    dstpath = op.join(opts.DATAPATH_SRC, f"{dataset}_{split}")
    os.makedirs(dstpath, exist_ok=True)
    drive_paths = drive_lister.list_drive_paths()
    if not drive_paths:
        print("[Failure] There is no drives in", drive_lister.srcpath)
        return

    num_drives = len(drive_paths)
    for i, drive_path in enumerate(drive_paths):
        data_reader = df.dataset_reader_factory(opts.get_raw_data_path(dataset), drive_path, dataset, split)
        data_reader.init_drive()
        num_frames = data_reader.num_frames()
        if num_frames == 0:
            print("this drive is EMPTY:", drive_path)
            continue

        data_paths = drive_lister.make_saving_paths(drive_path)
        image_path = data_paths[0]
        if op.isdir(image_path):
            print(f"this drive is already prepared, check this path completed: {image_path.replace(opts.DATAPATH_SRC, '')}")
            continue

        print(f"\n{'=' * 50}\n[load drive] [{i+1}/{num_drives}] drive path: {image_path}, # frames={num_frames}")
        snippet_maker.set_reader(data_reader)

        with PathManager(data_paths) as pm:
            for example_index in range(num_frames):
                example = snippet_maker.get_example(example_index)
                filename = data_reader.get_filename(example_index)
                mean_depth = save_example(example, filename, data_paths)
                print_progress_status(f"Progress: mean depth={mean_depth:0.3f}, "
                                      f"file={filename} {example_index}/{num_frames}")
            # if set_ok() was NOT excuted, the generated path is removed
            pm.set_ok()
        print("")
    print(f"===== Data preparation of {dataset}_{split} is done\n\n")


def save_example(example, filename, data_paths):
    image_path, pose_path, depth_path = data_paths
    frames = example["image"]
    filepath = op.join(image_path, f"{filename}.png")
    cv2.imwrite(filepath, frames)
    center_y, center_x = opts.get_shape("HW", 2)

    filepath = op.join(image_path, "intrinsic.txt")
    if not op.isfile(filepath):
        intrinsic = example["intrinsic"]
        print("intrinsic parameters\n", intrinsic)
        np.savetxt(filepath, intrinsic, fmt="%3.5f")

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
    if op.isdir(dstpath):
        print("!!! The validation set has already been created. To create validation set, "
              "remove validation set and try again")
        return

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
    print(f"===== Data preparation of {dataset}_val is done\n\n")


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
    prepare_datasets()

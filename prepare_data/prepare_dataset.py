import os
import os.path as op
from config import opts
from kitti_loader import KittiDataLoader

# TODO: odometry에 static frame 목록 만들기
'''
https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
def draw_flow(img, flow, step=16):
    ...

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
https://eehoeskrap.tistory.com/124
'''


def prepare_input_data():
    for split in ["train", "test"]:
        loader = KittiDataLoader(opts.RAW_DATASET_PATH, opts.DATASET, split)
        prepare_and_save_snippets(loader, split)


def prepare_and_save_snippets(loader, split):
    dstpath = op.join(opts.DATA_PATH, "srcdata", f"{opts.DATASET}_{split}")
    # mkdir(dstpath)

    for drive in loader.drive_list:
        print("drive:", drive)
        loader.load_drive(drive)
        dst_drive_path = ""     # TODO
        for snippet in loader.snippet_generator(opts.SNIPPET_LEN):
            index = snippet["index"]
            frames = snippet["frames"]
            # op.join(dst_drive_path, f"{index:06d}.png")

            poses = snippet["gt_poses"]
            # op.join(dst_drive_path, "poses", f"{index:06d}.txt")

            depth = snippet["gt_depth"]
            # op.join(dst_drive_path, "depths", f"{index:06d}.txt")

            intrinsic = snippet["intrinsic"]
            # op.join(dst_drive_path, "intrinsic.txt")



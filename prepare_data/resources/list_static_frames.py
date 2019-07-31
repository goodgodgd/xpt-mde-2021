import os.path as op
from config import opts
import pykitti


def list_kitti_odom_static_frames():
    for drive_id in range(22):
        drive_path = op.join(opts.RAW_DATASET_PATH)


if __name__ == "__main__":
    list_kitti_odom_static_frames()
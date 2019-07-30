import os.path as op
from glob import glob
import numpy as np
from config import opts


def create_scene_split_files(base_path):
    all_drives = glob(base_path + "/*/*")
    all_drives = [drive.replace(base_path, "").strip("/") for drive in all_drives if op.isdir(drive)]
    all_drives = [drive[11:].replace("_sync", "").replace("_drive_", " ") + "\n" for drive in all_drives]
    print("all drives:  ", len(all_drives), all_drives)
    with open("test_scenes_eigen.txt", "r") as fr:
        lines = fr.readlines()
        test_drives = [line.replace("_drive_", " ") for line in lines]
        print("test drives: ", len(test_drives), test_drives)
        train_drives = set(all_drives) - set(test_drives)
        train_drives = list(train_drives)
        train_drives.sort()
        print("train drives:", len(train_drives), train_drives)

        with open("../resources/kitti_raw_train_scenes.txt", "w") as fw:
            fw.writelines(train_drives)

        with open("../resources/kitti_raw_test_scenes.txt", "w") as fw:
            fw.writelines(test_drives)


def convert_static_frame_format():
    with open("static_frames.txt", "r") as fr:
        lines = fr.readlines()
        new_lines = []
        for line in lines:
            date, drive, frame = line.split(" ")
            new_lines.append(f"{date} {drive[17:21]} {frame}")

        with open("../resources/kitti_raw_static_frames.txt", "w") as fw:
            fw.writelines(new_lines)


def convert_test_frames_format():
    with open("test_files_eigen.txt", "r") as fr:
        lines = fr.readlines()
        new_lines = []
        for line in lines:
            date, drive, _, _, frame = line.split("/")
            new_lines.append(f"{date} {drive[17:21]} {frame[:-5]}\n")

        print(new_lines)
        with open("../resources/kitti_test_depth_frames.txt", "w") as fw:
            fw.writelines(new_lines)


def create_false_trajectories():
    odom_path = "/media/ian/IanPrivatePP/Datasets/kitti_odometry"
    for drive in range(11, 22):
        frame_pattern = op.join(odom_path, "sequences", f"{drive:02d}", "image_2", "*.png")
        print("file pattern", frame_pattern)
        num_frames = len(glob(frame_pattern))
        print("num_frames", num_frames)
        one_pose = np.concatenate([np.identity(3), np.zeros((3,1))], axis=1).reshape(-1)
        print(one_pose)
        poses = np.tile(one_pose, (num_frames, 1))
        print("shape:", poses.shape)
        print(poses[:5])
        np.savetxt(op.join(odom_path, "poses", f"{drive:02d}.txt"), poses, fmt="%.6e")


if __name__ == "__main__":
    raw_data_path = "/media/ian/iandata/datasets/kitti_raw_data"
    # create_scene_split_files(raw_data_path)
    # convert_static_frame_format()
    # convert_test_frames_format()
    create_false_trajectories()

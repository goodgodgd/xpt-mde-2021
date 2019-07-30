import os
from glob import glob
from config import opts


def list_raw_scenes(base_path):
    all_drives = glob(base_path + "/*/*")
    all_drives = [drive.replace(base_path, "").strip("/") for drive in all_drives if os.path.isdir(drive)]
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


def list_static_frames():
    with open("static_frames.txt", "r") as fr:
        lines = fr.readlines()
        new_lines = []
        for line in lines:
            date, drive, frame = line.split(" ")
            new_lines.append(f"{date} {drive[17:21]} {frame}")

        with open("../resources/kitti_static_frames.txt", "w") as fw:
            fw.writelines(new_lines)


def list_test_frames():
    with open("test_files_eigen.txt", "r") as fr:
        lines = fr.readlines()
        new_lines = []
        for line in lines:
            date, drive, _, _, frame = line.split("/")
            new_lines.append(f"{date} {drive[17:21]} {frame[:-5]}\n")

        print(new_lines)
        with open("../resources/kitti_test_depth_frames.txt", "w") as fw:
            fw.writelines(new_lines)


if __name__ == "__main__":
    raw_data_path = "/media/ian/iandata/datasets/kitti_raw_data"
    # list_raw_scenes(raw_data_path)
    # list_static_frames()
    list_test_frames()

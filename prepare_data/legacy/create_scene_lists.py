import os
from glob import glob
from config import opts


def list_raw_scenes(base_path):
    odo_data_path = "/media/ian/iandata/datasets/kitti_odometry"
    all_drives = glob(base_path + "/*/*")
    all_drives = [drive.replace(base_path, "").strip("/") for drive in all_drives if os.path.isdir(drive)]
    all_drives = [drive[11:].replace("_sync", "").replace("_drive_", " ") + "\n" for drive in all_drives]
    print("all drives:  ", len(all_drives), all_drives)
    with open("test_scenes_eigen.txt", "r") as f:
        lines = f.readlines()
        test_drives = [line.replace("_drive_", " ") for line in lines]
        print("test drives: ", len(test_drives), test_drives)
        train_drives = set(all_drives) - set(test_drives)
        train_drives = list(train_drives)
        train_drives.sort()
        print("train drives:", len(train_drives), train_drives)

        with open("../resources/kitti_raw_train_scenes.txt", "w") as f:
            f.writelines(train_drives)

        with open("../resources/kitti_raw_test_scenes.txt", "w") as f:
            f.writelines(test_drives)


if __name__ == "__main__":
    raw_data_path = "/media/ian/iandata/datasets/kitti_raw_data"
    list_raw_scenes(raw_data_path)

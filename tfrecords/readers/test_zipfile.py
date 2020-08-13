import zipfile
import tarfile
from PIL import Image
import cv2
import numpy as np
import json
from glob import glob
from utils.util_funcs import print_progress_status
from matplotlib import pyplot as plt


def count_images_from_zip(zip_pattern, image_suffix):
    print("==== zip pattern:", zip_pattern)
    zip_files = glob(zip_pattern)
    frame_counts = []
    for zip_name in zip_files:
        print("== zip file:", zip_name)
        zfile = zipfile.ZipFile(zip_name)
        filelist = zfile.namelist()
        filelist = [file for file in filelist if file.endswith(image_suffix)]
        filelist.sort()
        if len(filelist) < 100:
            continue

        frame_counts.append(len(filelist))
        print("image file list:", len(filelist))
        for i in range(0, len(filelist), len(filelist)//5):
            print(i, filelist[i])

        image_bytes = zfile.open(filelist[0])
        image = Image.open(image_bytes)
        image = np.array(image, np.uint8)
        print("image:", image.shape, image.dtype, np.max(image), np.min(image), np.median(image))
        cv2.imshow("image", image)
        image[image > 10000] = 2 ** 16 - 1
        cv2.imshow("depth", image)
        cv2.waitKey(0)

    frame_counts = np.array(frame_counts)
    total_frames = np.sum(frame_counts)
    print("frame_counts:", frame_counts)
    print("total_frames:", total_frames)
    print("\n\n")


def show_depth_from_zip(zip_pattern, image_suffix):
    print("==== zip pattern:", zip_pattern)
    zip_files = glob(zip_pattern)
    for zip_name in zip_files:
        print("== zip file:", zip_name)
        zfile = zipfile.ZipFile(zip_name)
        filelist = zfile.namelist()
        filelist = [file for file in filelist if file.endswith(image_suffix)]
        filelist.sort()
        if len(filelist) < 100:
            continue

        print("disparity files:", len(filelist))
        for i in range(0, len(filelist), len(filelist)//5):
            print(i, filelist[i])
            disp_bytes = zfile.open(filelist[i])
            disp = Image.open(disp_bytes)
            disp = np.array(disp, np.uint16).astype(np.float32)
            disp[disp > 0] = (disp[disp > 0] - 1) / 256.
            depth = np.copy(disp)
            # depth = baseline * focal length / disparity
            depth[disp > 0] = (0.209313 * 2262.52) / disp[disp > 0]
            depth[depth > 50] = 50
            print("depth:", i, np.min(depth[depth > 0]), np.max(depth[depth > 0]), np.median(depth[depth > 0]))
            depth_view = (depth / 50. * 256).astype(np.uint8)
            depth_view = cv2.applyColorMap(depth_view, cv2.COLORMAP_SUMMER)
            cv2.imshow("depth", depth_view)
            cv2.waitKey(0)


def open_text_from_zip():
    zfile = zipfile.ZipFile("/media/ian/IanBook/datasets/raw_zips/cityscapes/camera_trainextra.zip")
    filelist = zfile.namelist()
    filelist = [file for file in filelist if file.endswith(".json")]
    filelist.sort()
    print("text file list:", len(filelist))
    for i in range(0, len(filelist), 1000):
        print(i, filelist[i])

    contents = zfile.read(filelist[0])
    print("text contents", contents)
    params = json.loads(contents)
    print("params", type(params), params)


def convert_tar_to_vanilla_zip():
    print("\n==== convert_tar_to_vanilla_zip")
    tar_pattern = "/media/ian/IanBook/datasets/raw_zips/nuscenes/v1.0-trainval*.tar"

    tar_files = glob(tar_pattern)
    for ti, tar_name in enumerate(tar_files):
        if ti > 2:
            break
        print("==== tar file:", tar_name)
        tfile = tarfile.open(tar_name, 'r')
        zip_name = tar_name.replace(".tar", ".zip")
        zip_name = zip_name.replace("/nuscenes/", "/nuscenes_zip/")
        print("==== zip file:", zip_name)
        zfile = zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_STORED)

        for fi, intar_file in enumerate(tfile):
            if not intar_file.isfile():
                continue
            inzip_name = intar_file.name
            if "sweeps/CAM_FRONT/n" not in inzip_name:
                print_progress_status(f"== converting: (NOT FRONT) {ti} tar, {fi} file in tar, {inzip_name[:40]}")
                continue

            contents = tfile.extractfile(intar_file)
            contents = contents.read()
            zfile.writestr(inzip_name, contents)
            print_progress_status(f"== converting: (CAM_FRONT) {ti}-th tar, {fi}-th file in tar, {inzip_name[:40]}")

        print("\n==== close tar and zip")
        tfile.close()
        zfile.close()


if __name__ == "__main__":
    # zip_pattern = "/media/ian/IanBook/datasets/raw_zips/cityscapes/leftImg8bit_*.zip"
    # count_images_from_zip(zip_pattern, ".png")

    zip_pattern = "/media/ian/IanBook/datasets/raw_zips/cityscapes/disparity_*.zip"
    show_depth_from_zip(zip_pattern, ".png")

    # zip_pattern = "/media/ian/IanBook/datasets/raw_zips/driving_stereo/train-left-image/*.zip"
    # count_images_from_zip(zip_pattern, ".jpg")

    # open_text_from_zip()

    # convert_tar_to_vanilla_zip()
    # zip_pattern = "/media/ian/IanBook/datasets/raw_zips/nuscenes_zip/*.zip"
    # count_images_from_zip(zip_pattern, ".jpg")


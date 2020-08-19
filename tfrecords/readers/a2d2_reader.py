import os
import os.path as op
import zipfile
import tarfile
from PIL import Image
import cv2
import numpy as np
import json
from glob import glob
from utils.util_funcs import print_progress_status

# TODO: check staic frame 20180810150607_camera_frontcenter_000007746.png


def convert_tar_to_vanilla_zip():
    print("\n==== convert_tar_to_vanilla_zip")
    tar_pattern = "/media/ian/IanBook/datasets/raw_zips/a2d2/*.tar"
    tar_files = glob(tar_pattern)
    tar_files = [file for file in tar_files if "frontcenter" not in file]
    for ti, tar_name in enumerate(tar_files):
        print("====== open tar file:", op.basename(tar_name))
        tfile = tarfile.open(tar_name, 'r')
        filename = op.basename(tar_name).replace(".tar", ".zip")
        zip_name = op.join(op.dirname(tar_name), "zips", filename)
        if op.isfile(zip_name):
            print(f"{op.basename(zip_name)} already made!!")
            continue
        os.makedirs(op.dirname(zip_name), exist_ok=True)
        print("== zip file:", op.basename(zip_name))
        zfile = zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_STORED)

        for fi, tarinfo in enumerate(tfile):
            # if fi >= 100:
            #     break
            if not tarinfo.isfile():
                continue
            inzip_name = tarinfo.name
            contents = tfile.extractfile(tarinfo)
            contents = contents.read()
            zfile.writestr(inzip_name, contents)
            print_progress_status(f"== converting: tar: {ti}, file: {fi}, {inzip_name[-45:]}")

        tfile.close()
        zfile.close()


if __name__ == "__main__":
    convert_tar_to_vanilla_zip()

import os.path as op
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import settings
from config import opts
from tfrecords.tfrecord_reader import TfrecordReader
import utils.util_funcs as uf


def visualize_by_user_interaction():
    options = {"data_dir_name": "kitti_raw_test",
               "model_name": "vode_model",
               }

    print("\n===== Select evaluation options")

    print(f"Default options:")
    for key, value in options.items():
        print(f"\t{key} = {value}")
    print("\nIf you are happy with default options, please press enter")
    print("Otherwise, please press any other key")
    select = input()

    if select == "":
        print(f"You selected default options.")
    else:
        message = "Type 1 or 2 to specify dataset: 1) kitti_raw_test, 2) kitti_odom_test"
        ds_id = uf.input_integer(message, 1, 2)
        if ds_id == 1:
            options["data_dir_name"] = "kitti_raw_test"
        if ds_id == 2:
            options["data_dir_name"] = "kitti_odom_test"

        print("Type model_name: dir name under opts.DATAPATH_CKP and opts.DATAPATH_PRD")
        options["model_name"] = input()

    print("Prediction options:", options)
    visualize(**options)


def visualize(data_dir_name, model_name):
    depths = np.load(op.join(opts.DATAPATH_PRD, model_name, "depth.npy"))
    poses = np.load(op.join(opts.DATAPATH_PRD, model_name, "pose.npy"))
    print(f"depth shape: {depths.shape}, pose shape: {poses.shape}")
    tfrgen = TfrecordReader(op.join(opts.DATAPATH_TFR, data_dir_name), batch_size=1)
    dataset = tfrgen.get_dataset()
    fig = plt.figure()
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)

    for i, (x, y) in enumerate(dataset):
        image = tf.image.convert_image_dtype((x["image"] + 1.)/2., dtype=tf.uint8)
        image = image[0].numpy()

        depth = np.squeeze(depths[i], axis=-1)
        pose_snippet = poses[i]
        print("source frame poses w.r.t target (center) frame")
        print(pose_snippet)

        cv2.imshow("image", image)
        cv2.waitKey(1000)
        print("depth", depth.shape)
        plt.imshow(depth, cmap="viridis")
        plt.show()


if __name__ == "__main__":
    visualize_by_user_interaction()

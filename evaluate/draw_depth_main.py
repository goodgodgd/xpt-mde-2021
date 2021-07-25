import os.path as op
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import settings
from config import opts

MD1_FILE = "/media/ri-bear/IntHDD/vode_data/vode_0103/othermethod/monodepth/disparities_city2eigen_resnet/disparities_pp.npy"
MD2_FILE = "/media/ri-bear/IntHDD/vode_data/vode_0103/othermethod/monodepth2/mono+stereo_1024x320_eigen.npy"


def visualize_depth(ckpt_name, dataset_name="kitti_raw"):
    evaldata = np.load(op.join(opts.DATAPATH_PRD, ckpt_name, dataset_name + "_latest.npz"))
    datlen = evaldata["image"].shape[0]
    depth_md1 = 1. / (np.load(MD1_FILE) + 1e-6)
    depth_md2 = 1. / (np.load(MD2_FILE) + 1e-6)
    print("monodepth shape", depth_md1.shape, depth_md2.shape)
    # print("monodepth disp max", np.max(depth_md1), np.max(depth_md2))
    # print("monodepth dept max", np.max(1 / (depth_md1 + 1e-6)), np.max(1 / (depth_md2 + 1e-6)))

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=500)  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i in range(datlen):
        image = evaldata["image"][i]
        depth_gt = evaldata["depth_gt"][i]
        depth_fill = depth_gt.copy()
        for k in range(3):
            depth_fill = fill_zero_depth(depth_fill)
        depth_pr = evaldata["depth"][i]
        print("--- eval depth:", i)
        if i < 38:
            continue
        
        plt.rcParams["figure.figsize"] = (6, 10)
        show_images([image, depth_fill, depth_pr, depth_md1[i], depth_md2[i]])
        plt.subplots_adjust(top=0.99, bottom=0.01, left=0.1, right=0.9, hspace=-0.35)
        # timer.start()
        plt.show()


def fill_zero_depth(depth):
    if depth.ndim == 3:
        depth = depth[..., 0]
    # add 3x3 neighbor pixels
    nei_depths = np.zeros((depth.shape[0], depth.shape[1], 9), dtype=np.float32)
    nei_depths[:-1, :-1, 0] += depth[1:, 1:]
    nei_depths[:-1, :, 1] += depth[1:, :]
    nei_depths[:-1, 1:, 2] += depth[1:, :-1]
    nei_depths[:, :-1, 3] += depth[:, 1:]
    nei_depths[:, :, 4] += depth[:, :]
    nei_depths[:, 1:, 5] += depth[:, :-1]
    nei_depths[1:, :-1, 6] += depth[:-1, 1:]
    nei_depths[1:, :, 7] += depth[:-1, :]
    nei_depths[1:, 1:, 8] += depth[:-1, :-1]
    depth_new = np.sum(nei_depths, axis=-1)
    valid_cnt = np.sum(nei_depths > 0, axis=-1)
    depth_new = depth_new / (valid_cnt + 1e-6)
    depth_new[depth > 0] = depth[depth > 0]
    return depth_new


def show_images(images):
    for i, image in enumerate(images):
        ax = plt.subplot(len(images), 1, i+1)
        plt.imshow(image)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


if __name__ == "__main__":
    visualize_depth("vode28_static_comb", "kitti_raw")

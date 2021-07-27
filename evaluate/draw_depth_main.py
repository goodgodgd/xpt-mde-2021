import os
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
    pred_data = np.load(op.join(opts.DATAPATH_PRD, ckpt_name, dataset_name + "_latest.npz"))
    print("pred_data.files:", pred_data.files)
    images = pred_data["image"]
    datlen, height, width, _ = images.shape
    evaldata = dict()
    evaldata["depth_md1"] = 1. / (np.load(MD1_FILE) + 1e-6)
    evaldata["depth_md2"] = 1. / (np.load(MD2_FILE) + 1e-6)
    evaldata["depth_xpt"] = pred_data["depth"]
    dirname = op.join(opts.DATAPATH_EVL, ckpt_name, "comparison")
    os.makedirs(dirname, exist_ok=True)

    max_disp = 2.5

    for i in range(datlen):
        image = images[i]
        result = [image]

        for key, depths in evaldata.items():
            depth = depths[i]
            # print("eval data:", i, key, depth.shape)
            if key == "depth_gt":
                depth = fill_zero_depth(depth)
                depth = fill_zero_depth(depth)

            # disp = 1. / (depth + 1e-6)
            disp = depth.copy()
            disp[depth > 0] = 1 / depth[depth > 0]
            if disp.shape[:2] != (128, 512):
                disp = cv2.resize(disp, (512, 155))
                disp = disp[18:-9]

            scale = np.mean(disp[-50:, 128:-128])
            disp = disp / scale
            disp = np.clip(disp, 0, max_disp)
            disp = (disp / max_disp * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_MAGMA)
            result.append(disp_color)

        result = np.concatenate(result, axis=0)
        result[128:-2:128] = 0
        cv2.imwrite(op.join(dirname, f"compare_{i:03d}.png"), result)
        cv2.imshow("result", result)
        cv2.waitKey()


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

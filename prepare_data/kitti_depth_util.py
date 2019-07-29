# Mostly based on the code written by Clement Godard: 
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import os
import cv2
from collections import Counter


def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def save_gt_depths(depths, output_root):
    if not os.path.isdir(output_root):
        raise FileNotFoundError(output_root)

    save_path = os.path.join(output_root, "ground_truth", "depth")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for i, depth in enumerate(depths):
        filename = os.path.join(save_path, "{:06d}".format(i))
        np.save(filename, depth)


def save_pred_depths(depths, output_root, modelname):
    if not os.path.isdir(os.path.join(output_root, modelname)):
        raise FileNotFoundError()

    save_path = os.path.join(output_root, modelname, "depth")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    depths = np.concatenate(depths, axis=0)
    filename = os.path.join(save_path, "kitti_eigen_depth_predictions")
    np.save(filename, depths)
    print("predicted depths were saved!! shape=", depths.shape)


###############################################################################
# KITTI

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


def convert_disps_to_depths_kitti(gt_disparities, pred_disparities):
    gt_depths = []
    pred_depths = []
    pred_disparities_resized = []

    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_disp = pred_disparities[i]
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(pred_disp)

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths.append(pred_depth)
    return gt_depths, pred_depths, pred_disparities_resized


###############################################################################
# EIGEN

def read_file_data(files, data_root):
    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []
    num_probs = 0
    for i, filename in enumerate(files):
        filename = filename.split()[0]
        splits = filename.split('/')
        #         camera_id = filename[-1]   # 2 is left, 3 is right
        date = splits[0]
        im_id = splits[4][:10]
        vel = '{}/{}/velodyne_points/data/{}.bin'.format(splits[0], splits[1], im_id)
        imfile = os.path.join(data_root, filename)

        if os.path.isfile(imfile):
            gt_files.append(os.path.join(data_root, vel))
            gt_calib.append(os.path.join(data_root, date))
            im_sizes.append(cv2.imread(imfile).shape[:2])
            im_files.append(imfile)
            cams.append(2)
        else:
            num_probs += 1
            print('{} missing'.format(imfile))
    # print(num_probs, 'files missing')
    return gt_files, gt_calib, im_sizes, im_files, cams


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            # if value is array of numbers, not date time, convert to numpy array
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    # change: np.array(map(f, v)) -> np.array(list(map(f, v)))
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(velo_data, calib_dir, im_shape, cam='02', interp=False):
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_' + cam].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo_data = velo_data[velo_data[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo_data.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    if interp:
        # interpolate the depth map to fill in holes
        depth_interp = lin_interp(im_shape, velo_pts_im)
        return depth, depth_interp
    else:
        return depth


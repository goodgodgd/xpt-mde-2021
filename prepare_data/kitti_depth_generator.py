# Mostly based on the code written by Clement Godard: 
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import os
from collections import Counter


def generate_depth_map(velo_data, calib_dir, orig_shape, target_shape, cam='02', interp=False):
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

    """ CAUTION!
    orig_shape, target_shape: (height, width) 
    velo_pts_im[i, :] = (x, y)
    """
    orig_height, orig_width = orig_shape
    targ_height, targ_width = target_shape

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo_data.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    # rescale pixel points to target size
    velo_pts_im[:, 0] = velo_pts_im[:, 0] / orig_width * targ_width
    velo_pts_im[:, 1] = velo_pts_im[:, 1] / orig_height * targ_height

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < targ_width) & (velo_pts_im[:, 1] < targ_height)
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros(target_shape)
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
        depth_interp = lin_interp(target_shape, velo_pts_im)
        return depth, depth_interp
    else:
        return depth


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


# deprecated
def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

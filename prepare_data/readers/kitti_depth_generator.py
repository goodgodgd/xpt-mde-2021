import numpy as np
from collections import Counter


def generate_depth_map(velo_data, T_cam_velo, K_cam, orig_shape, target_shape):
    # remove all velodyne points behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance(0)
    velo_data = velo_data[velo_data[:, 0] >= 0, :].T    # (N, 4) => (4, N)
    velo_data[3, :] = 1
    velo_in_camera = np.dot(T_cam_velo, velo_data)      # => (3, N)

    """ CAUTION!
    orig_shape, target_shape: (height, width) 
    velo_data[i, :] = (x, y, z)
    """
    targ_height, targ_width = target_shape
    orig_height, orig_width = orig_shape
    # rescale intrinsic parameters to target image shape
    K_prj = K_cam.copy()
    K_prj[0, :] *= (targ_width / orig_width)    # fx, cx *= target_width / orig_width
    K_prj[1, :] *= (targ_height / orig_height)  # fy, cy *= target_height / orig_height

    # project the points to the camera
    velo_pts_im = np.dot(K_prj, velo_in_camera[:3])         # => (3, N)
    velo_pts_im[:2] = velo_pts_im[:2] / velo_pts_im[2:3]    # (u, v, z)

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[0] = np.round(velo_pts_im[0]) - 1
    velo_pts_im[1] = np.round(velo_pts_im[1]) - 1
    valid_x = (velo_pts_im[0] >= 0) & (velo_pts_im[0] < targ_width)
    valid_y = (velo_pts_im[1] >= 0) & (velo_pts_im[1] < targ_height)
    velo_pts_im = velo_pts_im[:, valid_x & valid_y]

    # project to image
    depth = np.zeros(target_shape)
    depth[velo_pts_im[1].astype(np.int), velo_pts_im[0].astype(np.int)] = velo_pts_im[2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    return depth


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1

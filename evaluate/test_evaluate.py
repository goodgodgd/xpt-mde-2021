import os
import os.path as op
import numpy as np

import settings
from config import opts
import evaluate.evaluate_main as ev
import utils.util_funcs as uf


def test_evaluate_pose():
    pose_pred0 = np.array([[1, 2, 3, 0, 0, 1.], [4, 5, 6, 0, 0, 1.5], [1, 2, 3, 0, 0, 1.], [4, 5, 6, 0, 0, 1.5]])
    pose_true = pose_pred0.copy()

    # scaling trajectory does not affect error
    pose_pred1 = pose_pred0.copy()
    pose_pred1[:, :3] = pose_pred1[:, :3] * 2
    print("predicted_pose\n", pose_pred1)
    pose_true = uf.pose_rvec2matr(pose_true)
    trj_err, rot_err = ev.evaluate_pose(pose_pred1, pose_true)
    print("trajectory error\n", trj_err)
    assert np.isclose(trj_err, 0).all()

    # change orientations
    pose_pred2 = pose_pred0.copy()
    rotation = np.array([0, -1, 1, 2])
    pose_pred2[:, 5] = pose_pred2[:, 5] + rotation
    print("predicted_pose\n", pose_pred2)
    trj_err, rot_err = ev.evaluate_pose(pose_pred2, pose_true)
    print("rotational error\n", rot_err)
    assert np.isclose(rot_err[1:], np.abs(rotation[[1, 0, 2, 3]])).all()

    # change orientation of origin
    pose_pred3 = pose_pred0.copy()
    rotation = np.array([1, 0, 0, 0])
    pose_pred3[:, 5] = pose_pred3[:, 5] + rotation
    print("predicted_pose\n", pose_pred3)
    trj_err, rot_err = ev.evaluate_pose(pose_pred3, pose_true)
    print("rotational error\n", rot_err)
    assert np.isclose(rot_err[1:], np.abs(rotation[0])).all()

    print("!!! test_evaluate_pose passed")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    test_evaluate_pose()


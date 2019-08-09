import sys
import numpy as np
import quaternion


def print_progress(count, is_total: bool = False):
    if is_total:
        # static variable in function
        print_progress.total = getattr(print_progress, 'last_hour', count)
    else:
        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {}/{}".format(count, print_progress.total)
        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    if count == print_progress.total:
        print("")


def pose_quat2mat(pose):
    t = np.expand_dims(pose[:3], axis=1)
    q = pose[3:]
    q = quaternion.from_float_array(q)
    # transpose: this rotation matrix is for "frame" rotation, not point rotation
    rot = quaternion.as_rotation_matrix(q).T
    mat = np.concatenate([rot, t], axis=1)
    mat = np.concatenate([mat, np.array([[0, 0, 0, 1]])], axis=0)
    return mat


def pose_mat2quat(pose):
    trans = pose[:3, 3]
    rot_mat = pose[:3, :3].T
    quat = quaternion.from_rotation_matrix(rot_mat)
    quat = quaternion.as_float_array(quat)
    pose_quat = np.concatenate([trans, quat], axis=0)
    return pose_quat


# ==================== tests ====================

def test_pose_quat2mat():
    mat = pose_quat2mat(np.array([0, 0, 0, 1, 0, 0, 0]))
    assert (np.isclose(np.identity(4), mat).all()), \
        f"[test_pose_quat2mat], np.identity(4) != \n{mat}"
    print("test_pose_quat2mat passed")


def test_convert_pose():
    pose_quat1 = np.array([1, 2, 3, np.cos(np.pi/3), 0, 0, np.sin(np.pi/3)])
    pose_mat = pose_quat2mat(pose_quat1)
    pose_quat2 = pose_mat2quat(pose_mat)

    quat1 = quaternion.from_float_array(pose_quat1[3:])
    quat2 = quaternion.from_float_array(pose_quat2[3:])
    rotm1 = quaternion.as_rotation_matrix(quat1)
    rotm2 = quaternion.as_rotation_matrix(quat2)
    print(f"convert pose quat1={quat1}, quat2={quat2}")

    assert (np.isclose(pose_quat1[:3], pose_quat2[:3]).all())
    assert (np.isclose(rotm1, rotm2).all())
    print("test_convert_pose passed")


def test():
    np.set_printoptions(precision=3, suppress=True)
    test_pose_quat2mat()
    test_convert_pose()


if __name__ == "__main__":
    test()

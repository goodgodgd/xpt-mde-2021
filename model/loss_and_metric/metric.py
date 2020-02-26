import tensorflow as tf
import utils.convert_pose as cp
import evaluate.eval_funcs as ef


def compute_metric_pose(pose_pred, pose_true_mat, stereo=False):
    """
    :param pose_pred: 6-DoF poses [batch, num_src, 6]
    :param pose_true_mat: 4x4 transformation matrix [batch, num_src, 4, 4]
    :param stereo: if stereo true, metric is evaluated in ABSOLUTE scale
    """
    pose_pred_mat = cp.pose_rvec2matr_batch(pose_pred)
    trj_err = ef.calc_trajectory_error_tensor(pose_pred_mat, pose_true_mat, stereo)
    rot_err = ef.calc_rotational_error_tensor(pose_pred_mat, pose_true_mat)
    return tf.reduce_mean(trj_err), tf.reduce_mean(rot_err)


def depth_error_metric(depth_pred, depth_true):
    """
    :param depth_pred: predicted depth [batch, height, width, 1]
    :param depth_true: ground truth depth [batch, height, width, 1]
    :return: depth error metric (scalar)
    """
    # flatten depths
    batch, height, width, _ = depth_pred.get_shape().as_list()
    depth_pred_vec = tf.reshape(depth_pred, (batch, height*width))
    depth_true_vec = tf.reshape(depth_true, (batch, height*width))

    # filter out zero depths
    depth_invalid_mask = tf.math.equal(depth_true_vec, 0)
    depth_pred_vec = tf.where(depth_invalid_mask, tf.constant(0, dtype=tf.float32), depth_pred_vec)
    depth_true_vec = tf.where(depth_invalid_mask, tf.constant(0, dtype=tf.float32), depth_true_vec)

    # normalize depths, [height*width, batch] / [batch] = [height*width, batch]
    depth_pred_vec = tf.transpose(depth_pred_vec) / tf.reduce_mean(depth_pred_vec, axis=1)
    depth_true_vec = tf.transpose(depth_true_vec) / tf.reduce_mean(depth_true_vec, axis=1)
    # [height*width, batch] -> [batch]
    depth_error = tf.reduce_mean(tf.abs(depth_pred_vec - depth_true_vec), axis=0)
    return depth_error

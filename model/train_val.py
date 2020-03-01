import tensorflow as tf
import numpy as np
import time

import utils.util_funcs as uf
from model.loss_and_metric.loss_factory import loss_factory
from model.loss_and_metric.metric import compute_metric_pose


class TrainValBase:
    def __init__(self, train_val_name, steps_per_epoch, stereo, optimizer=None):
        self.train_val_name = train_val_name
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.stereo = stereo
        self.weights = None

    def run_an_epoch(self, model, dataset):
        results = []
        depths = []
        compute_loss = loss_factory()
        # tf.data.Dataset object is reusable after a full iteration, check test_reuse_dataset()
        for step, features in enumerate(dataset):
            start = time.time()
            preds, loss, loss_by_type = self.run_a_batch(model, features, compute_loss, self.optimizer)
            batch_result, log_msg = merge_results(features, preds, loss, loss_by_type, self.stereo)
            mean_depths = get_center_depths(features, preds)
            uf.print_progress_status(f"    {self.train_val_name} {step}/{self.steps_per_epoch} steps, {log_msg}, "
                                     f"time={time.time() - start:1.4f}...")
            inspect_model(preds, step, self.steps_per_epoch)
            results.append(batch_result)
            depths.append(mean_depths)

        print("")
        # mean_result: mean of [all losses, trj_err, rot_err, weighted losses from various loss types]
        mean_result = np.array(results).mean(axis=0)
        # depths: [2, # frames in dataset]
        depths = np.concatenate(depths, axis=1)
        print(f"[{self.train_val_name} Epoch MEAN], result: loss={mean_result[0]:1.4f}, "
              f"trj_err={mean_result[1]:1.4f}, rot_err={mean_result[2]:1.4f}")
        return mean_result, depths

    def run_a_batch(self, model, features, compute_loss, optimizer):
        raise NotImplementedError()


class ModelTrainerGraph(TrainValBase):
    def __init__(self, steps_per_epoch, stereo, optimizer):
        super().__init__("Train (graph)", steps_per_epoch, stereo, optimizer)

    @tf.function
    def run_a_batch(self, model, features, compute_loss, optimizer):
        with tf.GradientTape() as tape:
            # NOTE! preds = {"depth_ms": ..., "pose": ...} = model(image)
            preds = model(features)
            loss_batch, loss_by_type = compute_loss(preds, features)

        grads = tape.gradient(loss_batch, model.trainable_weights())
        optimizer.apply_gradients(zip(grads, model.trainable_weights()))
        loss_mean = tf.reduce_mean(loss_batch)
        loss_by_type = tf.reduce_mean(loss_by_type, axis=1)
        """
        preds: {"pose": ..., "depth_ms":, ...}
        loss_mean: loss scalar that is averaged over all this epoch  
        loss_by_type: loss [loss types]
        """
        return preds, loss_mean, loss_by_type


class ModelTrainerEager(TrainValBase):
    def __init__(self, steps_per_epoch, stereo, optimizer):
        super().__init__("Train (eager)", steps_per_epoch, stereo, optimizer)

    def run_a_batch(self, model, features, compute_loss, optimizer):
        with tf.GradientTape() as tape:
            # NOTE! preds = {"depth_ms": ..., "pose": ...} = model(image)
            preds = model(features)
            loss_batch, loss_by_type = compute_loss(preds, features)

        grads = tape.gradient(loss_batch, model.trainable_weights())
        optimizer.apply_gradients(zip(grads, model.trainable_weights()))
        loss_mean = tf.reduce_mean(loss_batch)
        loss_by_type = tf.reduce_mean(loss_by_type, axis=1)
        self.grads = grads
        """
        preds: {"pose": ..., "depth_ms":, ...}
        loss_mean: loss scalar that is averaged over all this epoch  
        loss_by_type: loss [loss types]
        """
        return preds, loss_mean, loss_by_type


class ModelValidaterGraph(TrainValBase):
    def __init__(self, steps_per_epoch, stereo):
        super().__init__("Validate (graph)", steps_per_epoch, stereo)

    @tf.function
    def run_a_batch(self, model, features, compute_loss, optimizer):
        preds = model(features)
        loss_batch, loss_by_type = compute_loss(preds, features)
        loss_mean = tf.reduce_mean(loss_batch)
        loss_by_type = tf.reduce_mean(loss_by_type, axis=1)
        return preds, loss_mean, loss_by_type


class ModelValidaterEager(TrainValBase):
    def __init__(self, steps_per_epoch, stereo):
        super().__init__("Validate (eager)", steps_per_epoch, stereo)

    def run_a_batch(self, model, features, compute_loss, optimizer):
        preds = model(features)
        loss_batch, loss_by_type = compute_loss(preds, features)
        loss_mean = tf.reduce_mean(loss_batch)
        loss_by_type = tf.reduce_mean(loss_by_type, axis=1)
        return preds, loss_mean, loss_by_type


def merge_results(features, preds, loss, loss_by_type, stereo):
    trjerr, roterr = get_metric_pose(preds, features, stereo)
    batch_result = [loss.numpy(), trjerr, roterr] + loss_by_type.numpy().tolist()
    log_msg = f"loss = {loss.numpy():1.4f}, metric={trjerr:1.4f}, {roterr:1.4f}"
    return batch_result, log_msg


def get_center_depths(features, preds):
    pred_depth_ms = preds["depth_ms"]
    depth_pred = pred_depth_ms[0].numpy()
    batch, height, width, _ = depth_pred.shape
    if "depth_gt" in features:
        depth_true = features["depth_gt"].numpy()
    else:
        depth_true = np.zeros((batch, height, width, 1), np.float)
    xs, xe = width // 2 - 10, width // 2 + 10
    ys, ye = height // 4 * 3 - 10, height // 4 * 3 + 10

    depth_true = depth_true[:, ys:ye, xs:xe, :]
    mean_true = []
    for depth in depth_true:
        mean_d = depth[depth > 0].mean()
        mean_true.append(mean_d)
    mean_true = np.array(mean_true)
    mean_pred = np.mean(depth_pred[:, ys:ye, xs:xe, :], axis=(1, 2, 3))
    mean_depths = np.stack([mean_true, mean_pred], axis=0)
    """
    mean_depths: [2, batch] mean of true depths (row0) and predicted depths (row1)
    """
    return mean_depths


def get_metric_pose(preds, features, stereo):
    if "pose_gt" in features:
        trjerr, roterr = compute_metric_pose(preds['pose'], features['pose_gt'], stereo)
        return trjerr.numpy(), roterr.numpy()
    else:
        return 0, 0


def inspect_model(preds, step, steps_per_epoch):
    stride = steps_per_epoch // 5
    if step % stride > 0:
        return

    print("")
    print("depth0--", np.quantile(preds["depth_ms"][0].numpy(), np.arange(0.1, 1, 0.1)))
    print("dpconv0", np.quantile(preds["debug_out"][0].numpy(), np.arange(0.1, 1, 0.1)))
    print("upconv0", np.quantile(preds["debug_out"][1].numpy(), np.arange(0.1, 1, 0.1)))
    print("depth3--", np.quantile(preds["depth_ms"][3].numpy(), np.arange(0.1, 1, 0.1)))
    print("dpconv3", np.quantile(preds["debug_out"][2].numpy(), np.arange(0.1, 1, 0.1)))
    print("upconv3", np.quantile(preds["debug_out"][3].numpy(), np.arange(0.1, 1, 0.1)))
    print("pose_LR", preds["pose_LR"][0].numpy())

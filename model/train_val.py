import tensorflow as tf
import numpy as np
import pandas as pd
import time

import utils.util_funcs as uf
import utils.util_class as uc
import evaluate.eval_utils as eu
from model.model_util.distributer import DistributionStrategy, ReplicaOutputIntegrator


def train_val_factory(mode_sel, model, loss_object, steps_per_epoch, stereo, augmenter, optimizer):
    if mode_sel == "eager":
        trainer = ModelTrainer(model, loss_object, steps_per_epoch, stereo, augmenter, optimizer)
        validater = ModelValidater(model, loss_object, steps_per_epoch, stereo)
    elif mode_sel == "graph":
        trainer = ModelTrainerGraph(model, loss_object, steps_per_epoch, stereo, augmenter, optimizer)
        validater = ModelValidaterGraph(model, loss_object, steps_per_epoch, stereo)
    elif mode_sel == "distributed":
        trainer = ModelTrainerDistrib(model, loss_object, steps_per_epoch, stereo, augmenter, optimizer)
        validater = ModelValidaterDistrib(model, loss_object, steps_per_epoch, stereo)
    else:
        raise uc.WrongInputException(f"training mode '{mode_sel}' is NOT available")

    return trainer, validater


class TrainValBase:
    def __init__(self, model, loss_object, steps_per_epoch, stereo, augmenter=None, optimizer=None):
        self.model = model
        self.augmenter = augmenter
        self.loss_object = loss_object
        self.train_val_name = "train_val"
        self.steps_per_epoch = steps_per_epoch
        self.stereo = stereo
        self.optimizer = optimizer
        self.weights = None

    def set_name(self, name):
        self.train_val_name = name

    # tf.data.Dataset object is reusable after a full iteration, check test_reuse_dataset()
    def run_an_epoch(self, dataset):
        results = []
        with uc.DurationTime() as epoch_time:
            for step, features in enumerate(dataset):
                with uc.DurationTime() as step_time:
                    preds, loss, loss_by_type = self.run_a_batch(features)
                    batch_result, log_msg = merge_results(features, preds, loss, loss_by_type, self.stereo)
                uf.print_progress_status(f"    {self.train_val_name} {step}/{self.steps_per_epoch} steps, {log_msg}, "
                                         f"time={step_time.duration:1.4f}...")
                inspect_model(preds, features, step, self.steps_per_epoch)
                results.append(batch_result)

        print("")
        # list of dict -> dataframe -> mean: single row dataframe -> to_dict: dict of mean values
        results = pd.DataFrame(results)
        mean_results = results.mean(axis=0).to_dict()
        epoch_time = epoch_time.duration / 3600.  # second -> hour
        message = f"[{self.train_val_name} Epoch MEAN], result: "
        for key, val in mean_results.items():
            message += f"{key}={val:1.4f}, "
        print(message, "\n\n")
        return results, epoch_time

    def run_a_batch(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, steps_per_epoch, stereo, augmenter, optimizer):
        super().__init__(model, loss_object, steps_per_epoch, stereo, augmenter, optimizer)
        self.set_name("Train (eager)")

    def run_a_batch(self, features):
        return self.train_a_step(features)

    def train_a_step(self, features):
        features = self.augmenter(features)
        with tf.GradientTape() as tape:
            # preds = {"depth_ms": ..., "pose": ...} = model(image)
            preds = self.model(features)
            total_loss, loss_by_type = self.loss_object(preds, features)

        grads = tape.gradient(total_loss, self.model.trainable_weights())
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights()))
        """
        preds: {"pose": ..., "depth_ms":, ...}
        loss_mean: loss scalar that is averaged over all this epoch  
        loss_by_type: loss [loss types]
        """
        return preds, total_loss, loss_by_type


class ModelTrainerGraph(ModelTrainer):
    def __init__(self, model, loss_object, steps_per_epoch, stereo, augmenter, optimizer):
        super().__init__(model, loss_object, steps_per_epoch, stereo, augmenter, optimizer)
        self.set_name("Train (graph)")

    @tf.function
    def run_a_batch(self, features):
        return self.train_a_step(features)


class ModelTrainerDistrib(ModelTrainer):
    def __init__(self, model, loss_object, steps_per_epoch, stereo, augmenter, optimizer):
        super().__init__(model, loss_object, steps_per_epoch, stereo, augmenter, optimizer)
        self.strategy = DistributionStrategy.get_strategy()
        self.replica_integrator = ReplicaOutputIntegrator()
        self.set_name("Train (distributed)")

    @tf.function
    def run_a_batch(self, features):
        per_replica_results = self.strategy.run(self.train_a_step, args=(features,))
        per_replica_results = self.strategy.experimental_local_results(per_replica_results)
        return self.replica_integrator(per_replica_results)


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, steps_per_epoch, stereo):
        super().__init__(model, loss_object, steps_per_epoch, stereo)
        self.set_name("Validate (eager)")

    def run_a_batch(self, features):
        return self.validate_a_step(features)

    def validate_a_step(self, features):
        preds = self.model(features)
        total_loss, loss_by_type = self.loss_object(preds, features)
        return preds, total_loss, loss_by_type


class ModelValidaterGraph(ModelValidater):
    def __init__(self, model, loss_object, steps_per_epoch, stereo):
        super().__init__(model, loss_object, steps_per_epoch, stereo)
        self.set_name("Validate (graph)")

    @tf.function
    def run_a_batch(self, features):
        return self.validate_a_step(features)


class ModelValidaterDistrib(ModelValidater):
    def __init__(self, model, loss_object, steps_per_epoch, stereo):
        super().__init__(model, loss_object, steps_per_epoch, stereo)
        self.set_name("Validate (distributed)")
        self.strategy = DistributionStrategy.get_strategy()
        self.replica_integrator = ReplicaOutputIntegrator()

    @tf.function
    def run_a_batch(self, features):
        per_replica_results = self.strategy.run(self.validate_a_step, args=(features,))
        per_replica_results = self.strategy.experimental_local_results(per_replica_results)
        return self.replica_integrator(per_replica_results)


def merge_results(features, preds, loss, loss_by_type, stereo):
    batch_result = {"loss": loss.numpy()}
    log_msg = f"loss = {loss.numpy():1.4f}"
    if "pose" in preds:
        trj_abs_err, trj_rel_err, rot_err = get_pose_metric(preds, features)
        batch_result["trjabs"] = trj_abs_err
        batch_result["trjrel"] = trj_rel_err
        batch_result["roterr"] = rot_err
        log_msg += f", pose_err={trj_abs_err:1.4f}, {trj_rel_err:1.4f}, {rot_err:1.4f}"
    if "depth_ms" in preds:
        depth_abs_rel = get_depth_metric(features, preds)
        batch_result["deprel"] = depth_abs_rel
        log_msg += f", depth_err={depth_abs_rel:1.4f}"
        # compare center depths
        gtdepth, prdepth = get_center_depths(features, preds)
        batch_result["gtdepth"] = gtdepth[0]
        batch_result["prdepth"] = prdepth[0]
        # log_msg += f", gtdepth={gtdepth[0]:1.4f}, prdepth={prdepth[0]:1.4f}"
    loss_by_type = {key: loss.numpy() for key, loss in loss_by_type.items()}
    batch_result.update(loss_by_type)
    return batch_result, log_msg


def get_depth_metric(features, preds):
    pred_depth_ms = preds["depth_ms"]
    depth_pred = pred_depth_ms[0].numpy()
    batch, height, width, _ = depth_pred.shape
    if "depth_gt" in features:
        depth_true = features["depth_gt"].numpy()
    else:
        return 0

    depth_pred = depth_pred[..., 0]
    depth_true = depth_true[..., 0]
    """
    depth_pred, depth_true: [batch, height, width]
    depth_pred_val, depth_true_val: [batch, N]
    """
    metrics = []
    for depth_pr, depth_gt in zip(depth_pred, depth_true):
        depth_pr_val, depth_gt_val = eu.valid_depth_filter(depth_pr, depth_gt)
        abs_rel = np.mean(np.abs(depth_gt_val - depth_pr_val) / depth_gt_val)
        metrics.append(abs_rel)
    return np.mean(metrics)


def get_pose_metric(preds, features):
    if "pose_gt" in features:
        pose_eval = eu.PoseMetricTf()
        pose_eval.compute_pose_errors(preds['pose'], features['pose_gt'])
        return pose_eval.get_mean_pose_error()
    else:
        return 0, 0, 0


def get_center_depths(features, preds):
    pred_depth_ms = preds["depth_ms"]
    depth_pred = pred_depth_ms[0].numpy()
    batch, height, width, _ = depth_pred.shape
    if "depth_gt" in features:
        depth_true = features["depth_gt"].numpy()
    else:
        depth_true = np.ones((batch, height, width, 1), np.float)
    xs, xe = width // 2 - 10, width // 2 + 10
    ys, ye = height // 4 * 3 - 10, height // 4 * 3 + 10

    depth_true = depth_true[:, ys:ye, xs:xe, :]
    mean_true = []
    for depth in depth_true:
        if depth[depth > 0].sum() > 0:
            mean_d = depth[depth > 0].mean()
        else:
            mean_d = 0
        mean_true.append(mean_d)
    mean_true = np.array(mean_true)
    mean_pred = np.mean(depth_pred[:, ys:ye, xs:xe, :], axis=(1, 2, 3))
    """
    mean of true depths and predicted depths [batch]
    """
    return mean_true, mean_pred


def inspect_model(preds, features, step, steps_per_epoch):
    stride = steps_per_epoch // 3
    if step % stride > 0:
        return

    print("")
    if "depth_ms" in preds:
        print("depth0 ", np.quantile(preds["depth_ms"][0].numpy(), np.arange(0.1, 1, 0.1)))
        print("depth3 ", np.quantile(preds["depth_ms"][3].numpy(), np.arange(0.1, 1, 0.1)))
    if "debug_out" in preds:
        print("upconv0", np.quantile(preds["debug_out"][0].numpy(), np.arange(0.1, 1, 0.1)))
        print("upconv3", np.quantile(preds["debug_out"][1].numpy(), np.arange(0.1, 1, 0.1)))
    # flow: [batch, numsrc, height/4, width/4, 2] (4, 4, 32, 96, 2)
    if "flow_ms" in preds:
        print("flow0  ", np.quantile(preds["flow_ms"][0].numpy(), np.arange(0.1, 1, 0.1)))
    # pose: [batch, numsrc, 6]
    if "pose" in preds:
        print("pose_pr", preds["pose"][0, 0, :3].numpy(), preds["pose"][0, 1, :3].numpy())
    # pose_gt: [batch, numsrc, 4, 4]
    if "pose_gt" in features:
        print("pose_gt", features["pose_gt"][0, 0, :3, 3].numpy(), features["pose_gt"][0, 1, :3, 3].numpy())
    if "pose_LR" in preds:
        # pose: [batch, numsrc, 6]
        print("T_LR_pr", preds["pose_LR"][0, 0, :3].numpy(), preds["pose_LR"][0, 1, :3].numpy())
        # stereo_T_LR: [batch, 4, 4]
        print("T_LR_gt", features["stereo_T_LR"][0, :3, 3].numpy(), features["stereo_T_LR"][0, :3, 3].numpy())

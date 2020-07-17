import tensorflow as tf
import numpy as np
import time

import utils.util_funcs as uf
import utils.util_class as uc
from model.loss_and_metric.metric import compute_metric_pose
from model.model_util.distributer import DistributionStrategy, ReplicaOutputIntegrator


def train_val_factory(mode_sel, model, loss_object, steps_per_epoch, stereo, optimizer):
    if mode_sel == "eager":
        trainer = ModelTrainer(model, loss_object, steps_per_epoch, stereo, optimizer)
        validater = ModelValidater(model, loss_object, steps_per_epoch, stereo)
    elif mode_sel == "graph":
        trainer = ModelTrainerGraph(model, loss_object, steps_per_epoch, stereo, optimizer)
        validater = ModelValidaterGraph(model, loss_object, steps_per_epoch, stereo)
    elif mode_sel == "distributed":
        trainer = ModelTrainerDistrib(model, loss_object, steps_per_epoch, stereo, optimizer)
        validater = ModelValidaterDistrib(model, loss_object, steps_per_epoch, stereo)
    else:
        raise uc.WrongInputException(f"training mode '{mode_sel}' is NOT available")

    return trainer, validater


class TrainValBase:
    def __init__(self, train_val_name, model, loss_object, steps_per_epoch, stereo, optimizer=None):
        self.model = model
        self.loss_object = loss_object
        self.train_val_name = train_val_name
        self.steps_per_epoch = steps_per_epoch
        self.stereo = stereo
        self.optimizer = optimizer
        self.weights = None

    def run_an_epoch(self, dataset):
        results = []
        depths = []

        # tf.data.Dataset object is reusable after a full iteration, check test_reuse_dataset()
        for step, features in enumerate(dataset):
            start = time.time()
            preds, loss, loss_by_type = self.run_a_batch(features)
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

    def run_a_batch(self, features):
        raise NotImplementedError()


class ModelTrainer(TrainValBase):
    def __init__(self, model, loss_object, steps_per_epoch, stereo, optimizer):
        super().__init__("Train (graph)", model, loss_object, steps_per_epoch, stereo, optimizer)

    def run_a_batch(self, features):
        return self.train_a_step(features)

    def train_a_step(self, features):
        with tf.GradientTape() as tape:
            # NOTE! preds = {"depth_ms": ..., "pose": ...} = model(image)
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
    def __init__(self, model, loss_object, steps_per_epoch, stereo, optimizer):
        super().__init__(model, loss_object, steps_per_epoch, stereo, optimizer)

    @tf.function
    def run_a_batch(self, features):
        return self.train_a_step(features)


class ModelTrainerDistrib(ModelTrainer):
    def __init__(self, model, loss_object, steps_per_epoch, stereo, optimizer):
        super().__init__(model, loss_object, steps_per_epoch, stereo, optimizer)
        self.strategy = DistributionStrategy.get_strategy()
        self.replica_integrator = ReplicaOutputIntegrator()

    @tf.function
    def run_a_batch(self, features):
        per_replica_results = self.strategy.run(self.train_a_step, args=(features,))
        per_replica_results = self.strategy.experimental_local_results(per_replica_results)
        return self.replica_integrator(per_replica_results)


class ModelValidater(TrainValBase):
    def __init__(self, model, loss_object, steps_per_epoch, stereo):
        super().__init__("Validate (graph)", model, loss_object, steps_per_epoch, stereo)

    def run_a_batch(self, features):
        return self.validate_a_step(features)

    def validate_a_step(self, features):
        preds = self.model(features)
        total_loss, loss_by_type = self.loss_object(preds, features)
        return preds, total_loss, loss_by_type


class ModelValidaterGraph(ModelValidater):
    def __init__(self, model, loss_object, steps_per_epoch, stereo):
        super().__init__(model, loss_object, steps_per_epoch, stereo)

    @tf.function
    def run_a_batch(self, features):
        return self.validate_a_step(features)


class ModelValidaterDistrib(ModelValidater):
    def __init__(self, model, loss_object, steps_per_epoch, stereo):
        super().__init__(model, loss_object, steps_per_epoch, stereo)
        self.strategy = DistributionStrategy.get_strategy()
        self.replica_integrator = ReplicaOutputIntegrator()

    @tf.function
    def run_a_batch(self, features):
        per_replica_results = self.strategy.run(self.validate_a_step, args=(features,))
        per_replica_results = self.strategy.experimental_local_results(per_replica_results)
        return self.replica_integrator(per_replica_results)


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
        depth_true = np.ones((batch, height, width, 1), np.float)
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
    print("depth0 ", np.quantile(preds["depth_ms"][0].numpy(), np.arange(0.1, 1, 0.1)))
    print("dpconv0", np.quantile(preds["debug_out"][0].numpy(), np.arange(0.1, 1, 0.1)))
    print("upconv0", np.quantile(preds["debug_out"][1].numpy(), np.arange(0.1, 1, 0.1)))
    print("depth3 ", np.quantile(preds["depth_ms"][3].numpy(), np.arange(0.1, 1, 0.1)))
    print("dpconv3", np.quantile(preds["debug_out"][2].numpy(), np.arange(0.1, 1, 0.1)))
    print("upconv3", np.quantile(preds["debug_out"][3].numpy(), np.arange(0.1, 1, 0.1)))
    if "pose_LR" in preds:
        print("pose_LR", preds["pose_LR"][0].numpy())

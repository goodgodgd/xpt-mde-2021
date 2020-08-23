import tensorflow as tf
from config import opts


class DistributionStrategy:
    strategy = None

    @classmethod
    def get_strategy(cls):
        if (cls.strategy is None) and (opts.TRAIN_MODE == "distributed"):
            cls.strategy = tf.distribute.MirroredStrategy()
            opts.BATCH_SIZE = cls.strategy.num_replicas_in_sync * opts.PER_REPLICA_BATCH
            print(f"[DistributionStrategy] number of devices: {cls.strategy.num_replicas_in_sync}")
            print(f"[DistributionStrategy] global batch size: {opts.BATCH_SIZE}")
        return cls.strategy


class StrategyScope:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        if opts.TRAIN_MODE == "distributed":
            print("[StrategyScope]", self.func.__name__)
            strategy = DistributionStrategy.get_strategy()
            with strategy.scope():
                return self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)


class StrategyDataset:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        if opts.TRAIN_MODE == "distributed":
            print("[StrategyDataset]", self.func.__name__, *args)
            strategy = DistributionStrategy.get_strategy()
            dataset, steps = self.func(*args, **kwargs)
            dist_dataset = strategy.experimental_distribute_dataset(dataset)
            return dist_dataset, steps
        else:
            return self.func(*args, **kwargs)


class ReplicaOutputIntegrator:
    def __call__(self, per_replica_results):
        predictions = []
        loss_mean = []
        loss_by_type = []
        for replica_result in per_replica_results:
            predictions.append(replica_result[0])
            loss_mean.append(replica_result[1])
            loss_by_type.append(replica_result[2])

        predictions = self.integrate_predictions(predictions)
        loss_mean = self.integrate_scalar_loss(loss_mean)
        loss_by_type = self.integrate_dict_scalars(loss_by_type)
        return predictions, loss_mean, loss_by_type

    def integrate_predictions(self, replica_results):
        # replica_results: list of {key: value} for each replica
        # print("integrate dict:", replica_results)
        gather_outputs = {key: [] for key in replica_results[0]}
        for data in replica_results:
            for key, value in data.items():
                gather_outputs[key].append(value)
        # print("gather_outputs:", gather_outputs)

        # gather_outputs: {key: [replica data1, replica data2, ...]}
        integ_outputs = dict()
        for key, value in gather_outputs.items():
            if isinstance(value[0], list):
                integ_outputs[key] = self.integrate_replica_tensor_list(value)
            else:
                integ_outputs[key] = tf.concat(value, axis=0)
        # print("integ outputs", integ_outputs)
        return integ_outputs

    def integrate_replica_tensor_list(self, replica_results):
        # replica_results: list of [each replica results]
        outputs = [[] for i in range(len(replica_results[0]))]
        for per_replica_data in replica_results:
            for i, tensor in enumerate(per_replica_data):
                outputs[i].append(tensor)

        # outputs: list of the same type and shape data for each replica
        for i, across_replica_data in enumerate(outputs):
            outputs[i] = tf.concat(across_replica_data, axis=0)
        return outputs

    def integrate_scalar_loss(self, replica_results):
        # replica_results: list of loss_mean(scalar) for each replica
        result = replica_results[0] + replica_results[1]
        return tf.reduce_sum(replica_results)

    def integrate_dict_scalars(self, replica_results):
        # replica_results: list of loss_by_type {key: value} for each replica
        gather_outputs = {key: [] for key in replica_results[0]}
        for data in replica_results:
            for key, value in data.items():
                gather_outputs[key].append(value)
        # print("gather_outputs:", gather_outputs)

        integ_outputs = dict()
        for key, value in gather_outputs.items():
            integ_outputs[key] = tf.reduce_sum(value)
        # print("integ outputs", integ_outputs)
        return integ_outputs


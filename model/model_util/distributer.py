import tensorflow as tf
from config import opts


class DistributionStrategy:
    strategy = None

    @classmethod
    def get_strategy(cls):
        if cls.strategy is None:
            cls.strategy = tf.distribute.MirroredStrategy()
            print(f"[DistributionStrategy] Number of devices: {cls.strategy.num_replicas_in_sync}")
        return cls.strategy


class StrategyScope:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        if opts.USE_MULTI_GPU:
            strategy = DistributionStrategy.get_strategy()
            with strategy.scope():
                return self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)


class StrategyDataset:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        if opts.USE_MULTI_GPU:
            strategy = DistributionStrategy.get_strategy()
            datset, steps = self.func(*args, **kwargs)
            dist_dataset = strategy.experimental_distribute_dataset(datset)
            return dist_dataset, steps
        else:
            return self.func(*args, **kwargs)


class StrategyDis:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        if opts.USE_MULTI_GPU:
            strategy = DistributionStrategy.get_strategy()
            datset, steps = self.func(*args, **kwargs)
            dist_dataset = strategy.experimental_distribute_dataset(datset)
            return dist_dataset, steps
        else:
            return self.func(*args, **kwargs)


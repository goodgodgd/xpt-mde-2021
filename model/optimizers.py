import tensorflow as tf
from utils.util_class import WrongInputException

# TODO: make a optimizer policy class to change learning rates according to epochs


def optimizer_factory(opt_name, basic_lr, epoch=0):
    if opt_name == "adam_constant":
        return tf.optimizers.Adam(learning_rate=basic_lr)
    else:
        raise WrongInputException(f"{opt_name} is NOT an available model name")

import tensorflow as tf
from config import opts


def shape_check_real(func):
    def decorator(*args, **kwargs):
        print("@shape_check", func.__name__)
        for i, arg in enumerate(args):
            print_tensor_shape(arg, i, 'input')

        out = func(*args, **kwargs)

        print_tensor_shape(out, 0, f"{func.__name__} output")
        return out
    return decorator


def print_tensor_shape(tensor, index, name):
    if isinstance(tensor, tf.Tensor):
        print(f"  {name} {index}:", tensor.get_shape().as_list())
    elif isinstance(tensor, list):
        for k, val in enumerate(tensor):
            if isinstance(val, tf.Tensor):
                print(f"  {name} {index}-{k} in list:", val.get_shape().as_list())
            else:
                print(f"  {name} {index}-{k} is no tensor")
                break
    elif isinstance(tensor, dict):
        for key, val in tensor.items():
            if isinstance(val, tf.Tensor):
                print(f"  {name} {index}-{key} in dict:", val.get_shape().as_list())
            else:
                print(f"  {name} {index}-{key} is no tensor")
                break
    else:
        print(f"  {name} {index} is no tensor")


def shape_check_dummy(func):
    def decorator(*args, **kwargs):
        return func(*args, **kwargs)
    return decorator


"""
If you like to check in-out tensor shapes, set ENABLE_SHAPE_DECOR as True
"""
if opts.ENABLE_SHAPE_DECOR:
    shape_check = shape_check_real
else:
    shape_check = shape_check_dummy

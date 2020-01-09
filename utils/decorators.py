import tensorflow as tf
from config import opts


class ShapeCheckReal:
    def __init__(self, f):
        self.func = f
        self.name = f.__name__

    def __call__(self, *args, **kwargs):
        print("@ShapeCheck", self.name)
        for i, arg in enumerate(args):
            self.print_tensor_shape(arg, i, 'input')

        out = self.func(*args, **kwargs)

        self.print_tensor_shape(out, 0, f"{self.name} output")
        return out

    def print_tensor_shape(self, tensor, index, name):
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


class ShapeCheckDummy:
    def __init__(self, f):
        self.func = f
        self.name = f.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


"""
In case you like to check in-out tensor shapes, inherit "ShapeCheckReal"
otherwise, you can turn it off by inherinting "ShapeCheckDummy"
"""
if opts.ENABLE_SHAPE_DECOR:
    ShapeCheck = ShapeCheckReal
else:
    ShapeCheck = ShapeCheckDummy

import tensorflow as tf


class InOutShapeReal:
    def __init__(self, f):
        self.func = f
        self.name = f.__name__

    def __call__(self, *args, **kwargs):
        print("Function", self.name)
        for i, arg in enumerate(args):
            if isinstance(arg, tf.Tensor):
                print(f"  input arg {i}:", arg.get_shape().as_list())
            elif isinstance(arg, list) and isinstance(arg[0], tf.Tensor):
                print(f"  input arg {i} in list:", arg[0].get_shape().as_list())
            elif isinstance(arg, dict):
                val = list(arg.values())[0]
                if isinstance(val, tf.Tensor):
                    print(f"  input arg {i} in dict:", val.get_shape().as_list())
                else:
                    print(f"  input arg {i} is no tensor")
            else:
                print(f"  input arg {i} is no tensor")

        out = self.func(*args, **kwargs)

        if isinstance(out, tf.Tensor):
            print(f"[{self.name}] output shape:", out.get_shape().as_list())
        elif isinstance(out, list) and isinstance(out[0], tf.Tensor):
            print(f"[{self.name}] output shape:", out[0].get_shape().as_list())
        else:
            print(f"[{self.name}] output is no tensor")
        return out


class InOutShapeDummy:
    def __init__(self, f):
        self.func = f
        self.name = f.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


"""
In case you like to check in-out tensor shapes, inherit "InOutShapeReal"
otherwise, you can turn it off by inherinting "InOutShapeDummy"
"""
class InOutShape(InOutShapeReal):
    def __init__(self, f):
        super().__init__(f)

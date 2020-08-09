import numpy as np
import tensorflow as tf


class Serializer:
    def __call__(self, example_dict):
        features = self.convert_to_feature(example_dict)
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=features)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=features)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

    def convert_to_feature(self, example_dict):
        features = dict()
        for key, value in example_dict.items():
            if isinstance(value, np.ndarray):
                features[key] = self._bytes_feature(value.tostring())
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._int64_feature(value)
            else:
                assert 0, f"[convert_to_feature] Wrong data type: {type(value)}"
        return features

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def inspect_properties(example):
    config = dict()
    for key, value in example.items():
        config[key] = read_data_config(value)
    return config


def read_data_config(value):
    parse_type = ""
    decode_type = ""
    shape = ()
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            decode_type = "tf.uint8"
        elif value.dtype == np.float32:
            decode_type = "tf.float32"
        else:
            assert 0, f"[FeederBase] Wrong numpy type: {value.dtype}"
        parse_type = "tf.string"
        shape = list(value.shape)
    elif isinstance(value, int):
        parse_type = "tf.int64"
        shape = None
    else:
        assert 0, f"[FeederBase] Wrong type: {type(value)}"

    return {"parse_type": parse_type, "decode_type": decode_type, "shape": shape}



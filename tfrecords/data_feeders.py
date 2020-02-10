import numpy as np
import tensorflow as tf
from utils.util_class import WrongInputException


class FeederBase:
    def __init__(self):
        self.parse_type = ""
        self.decode_type = ""
        self.shape = []
    
    def __len__(self):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()

    def convert_to_feature(self, value):
        raise NotImplementedError()

    def set_type_and_shape(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8:
                self.decode_type = "tf.uint8"
            elif value.dtype == np.float32:
                self.decode_type = "tf.float32"
            else:
                raise WrongInputException(f"[FeederBase] Wrong numpy type: {value.dtype}")
            self.parse_type = "tf.string"
            self.shape = list(value.shape)

        elif isinstance(value, int):
            self.parse_type = "tf.int64"
            self.shape = None

        else:
            raise WrongInputException(f"[FeederBase] Wrong type: {type(value)}")

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


class FileFeeder(FeederBase):
    def __init__(self, file_list, file_reader):
        super().__init__()
        self.files = file_list
        self.file_reader = file_reader
        self.idx = -1
        value = self.file_reader(self.files[0])
        self.set_type_and_shape(value)

    def __len__(self):
        return len(self.files)

    def get_next(self):
        self.idx = self.idx + 1
        assert self.idx < len(self), f"[FileFeeder] index error: {self.idx} >= {len(self)}"
        onedata = self.file_reader(self.files[self.idx])
        return self.convert_to_feature(onedata)

    def convert_to_feature(self, value):
        raise NotImplementedError()


class NpyFeeder(FileFeeder):
    def __init__(self, file_list, file_reader):
        super().__init__(file_list, file_reader)

    def convert_to_feature(self, value):
        return self._bytes_feature(value.tostring())


class ConstArrayFeeder(FeederBase):
    def __init__(self, data, size):
        super().__init__()
        self.data = data
        self.size = size
        self.idx = -1
        self.set_type_and_shape(data)

    def __len__(self):
        return self.size

    def get_next(self):
        self.idx = self.idx + 1
        assert self.idx < len(self), f"[FileFeeder] index error: {self.idx} >= {len(self)}"
        return self.convert_to_feature(self.data)

    def convert_to_feature(self, value):
        return self._bytes_feature(value.tostring())

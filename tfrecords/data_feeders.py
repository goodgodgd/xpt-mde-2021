import numpy as np
import tensorflow as tf


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
            elif value.dtype == np.float64:
                self.decode_type = "tf.float64"
            else:
                print("numpy type:", value.dtype)
                raise TypeError()
            self.parse_type = "tf.string"
            self.shape = list(value.shape)

        elif isinstance(value, int):
            self.parse_type = "tf.int64"
            self.shape = None

        else:
            print("type:", type(value))
            raise TypeError()

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
        if self.idx >= len(self.files):
            raise IndexError()

        onedata = self.file_reader(self.files[self.idx])
        if onedata is None:
            raise FileNotFoundError(self.files[self.idx])

        # wrap a single raw data as tf.train.Features()
        features = dict()
        return self.convert_to_feature(onedata)

    def convert_to_feature(self, value):
        raise NotImplementedError()


class NpyFeeder(FileFeeder):
    def __init__(self, file_list, file_reader):
        super().__init__(file_list, file_reader)

    def convert_to_feature(self, value):
        return self._bytes_feature(value.tostring())


class ConstInt64Feeder(FeederBase):
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
        if self.idx >= self.size:
            raise IndexError()
        
        return self.convert_to_feature(self.data)

    def convert_to_feature(self, value):
        return self._int64_feature(value)


# ==================================================

def test():
    mat = np.identity(4)
    print(mat.dtype)
    feeder = NpyFeeder("test", 2, 3)
    feature = feeder.convert_to_feature(mat)
    print(feature)
    print(feature.value)
    value = tf.io.decode_raw(feature, tf.float64)


if __name__ == "__main__":
    test()

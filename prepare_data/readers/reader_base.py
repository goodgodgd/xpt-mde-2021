
class DataReaderBase:
    def __init__(self, base_path, drive_path, stereo=False):
        """
        :param base_path: root path of dataset
        :param drive_path: sequence drectory path
        when 'stereo' is True, 'get_xxx' function returns two data in tuple
        """
        self.base_path = base_path
        self.drive_path = drive_path
        self.stereo = stereo
        self.split = ""
        self.frame_count = [0, 0]
        self.frame_names = []
        self.frame_indices = []
        self.intrinsic = None
        self.T_left_right = None
        self.static_frames = []

    """
    Public methods used outside this class
    """
    def init_drive(self):
        """
        reset variables for a new sequence like intrinsic, extrinsic, and last index
        """
        raise NotImplementedError()

    def list_frames(self, drive_path):
        """
        :param drive_path: path to data of a drive
        :return: list of frame names and indices
        """
        raise NotImplementedError()

    def num_frames(self):
        """
        :return: number of frames of the drive
        """
        raise NotImplementedError()

    def get_image(self, index):
        """
        :return: indexed image in the current sequence
        """
        raise NotImplementedError()

    def get_quat_pose(self, index):
        """
        :return: indexed pose in a vector [position, quaternion] in the current sequence
        """
        raise NotImplementedError()

    def get_depth_map(self, index, raw_img_shape=None, target_shape=None):
        """
        :return: indexed pose in a vector [position, quaternion] in the current sequence
        """
        raise NotImplementedError()

    def get_intrinsic(self):
        """
        :return: camera projection matrix in the current sequence
        """
        raise NotImplementedError()

    def get_stereo_extrinsic(self):
        """
        :return: stereo extrinsic pose that transforms point in right frame into left frame
        """
        raise NotImplementedError()

    def get_filename(self, example_index):
        """
        :return: indexed frame file name
        """
        raise NotImplementedError()

    def get_frame_index(self, example_index):
        """
        :return: indexed frame file name
        """
        raise NotImplementedError()

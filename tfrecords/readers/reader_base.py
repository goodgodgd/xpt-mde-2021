
class DataReaderBase:
    def __init__(self, split):
        """
        when 'stereo' is True, 'get_xxx' function returns two data in tuple
        """
        self.split = split
        self.frame_names = []
        self.intrinsic = None
        self.T_left_right = None

    """
    Public methods used outside this class
    """
    def init_drive(self, drive_path):
        """
        :param drive_path: path to data of a drive
        reset variables for a new sequence like intrinsic, extrinsic, and last index
        """
        raise NotImplementedError()

    def num_frames_(self):
        """
        :return: number of frames of the drive
        """
        raise NotImplementedError()

    def get_range_(self):
        """
        :return: range object for frame index
        """
        raise NotImplementedError()

    def get_image(self, index, right=False):
        """
        :return: 'undistorted' indexed image in the current sequence
        """
        raise NotImplementedError()

    def get_pose(self, index, right=False):
        """
        :return: indexed pose in matrix format
        """
        raise NotImplementedError()

    def get_point_cloud(self, index, right=False):
        """
        :return: point cloud in standard camera frame (X=right, Y=down, Z=front)
        """
        raise NotImplementedError()

    def get_depth(self, index, srcshape_hw, dstshape_hw, intrinsic, right=False):
        """
        :return: indexed pose in a vector [position, quaternion] in the current sequence
        """
        raise NotImplementedError()

    def get_intrinsic(self, index=0, right=False):
        """
        :return: camera projection matrix in the current sequence
        """
        raise NotImplementedError()

    def get_stereo_extrinsic(self, index=0):
        """
        :return: stereo extrinsic pose that transforms point in right frame into left frame
        """
        raise NotImplementedError()

    def get_filename(self, index):
        """
        :return: indexed frame file name
        """
        return None

    def index_to_id(self, index):
        """
        :return: convert index of self.frame_names to frame id
        """
        # normally id is the same as index
        return index




class TrainException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class NoDataException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class TrainException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class WrongInputException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

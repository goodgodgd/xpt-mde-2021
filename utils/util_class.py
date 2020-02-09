import os
import shutil


class TrainException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class WrongInputException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class PathManager:
    def __init__(self, paths):
        self.paths = paths
        self.safe_exit = False

    def __enter__(self):
        for path in self.paths:
            if path is not None:
                os.makedirs(path, exist_ok=True)
        return self

    def set_ok(self):
        self.safe_exit = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.safe_exit is False:
            print("[PathManager] the process is NOT ended properly, remove the working paths")
            for path in self.paths:
                if path is not None and os.path.isdir(path):
                    print("    remove:", path)
                    shutil.rmtree(path)
            # to ensure the process stop here
            assert False

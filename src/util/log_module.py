# log module
import time

from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from functools import wraps

from util.command_option import get_version


def create_main_logger(version, mode="w"):
    logger_name = version + "main"
    log_filepath = Path(__file__).parents[2] / "log" / "main" / "{}.log".format(version)
    formatter = Formatter("[%(levelname)s]b %(asctime)s >>\t%(message)s")
    __create_logger(logger_name, log_filepath, formatter, mode)


def create_train_logger(version, mode="w"):
    logger_name = version + "train"
    log_filepath = Path(__file__).parents[2] / "log" / "train" / "{}.tsv".format(version)
    formatter = Formatter("%(message)s")
    __create_logger(logger_name, log_filepath, formatter, mode)


def __create_logger(logger_name, log_filepath, formatter, mode):
    Path.mkdir(log_filepath.parents[0], exist_ok=True, parents=True)

    logger = getLogger(logger_name)
    logger.setLevel(DEBUG)

    file_handler = FileHandler(log_filepath, mode=mode)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def get_main_logger(version):
    return getLogger(version + "main")


def get_train_logger(version):
    return getLogger(version + "train")


def stop_watch(*dargs, **dkargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            version = get_version()
            method_name = dargs[0]
            start = time.time()

            get_main_logger(version).info(f"====>> start  {method_name}")

            result = func(*args, **kargs)
            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            get_main_logger(version).info(f"<<==== finish {method_name}: [elapsed time] >> {hour:0>2}:{minits:0>2}:{sec:0>2}")
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    print(Path(__file__).parents[2].resolve())
    log_path = Path(__file__).parents[2] / "log" / "main" / "{}.log".format("0000")
    print(log_path.parents[0].resolve())

import os


def is_kagglekernel():
    return os.environ["HOME"] == "/tmp"

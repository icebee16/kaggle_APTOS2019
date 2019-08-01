import yaml
from pathlib import Path

import numpy as np

import torch

from util.command_option import get_version


def inference():
    pass


def load_config(self):
    """
    Loading yaml file.

    Returns
    -------
    config : dict
        information of process condition.
    """
    version = get_version()
    config_dir = Path(__file__).parents[1] / "config"
    config_file_list = list(config_dir.glob(f"{version}*.yml"))

    if len(config_file_list) > 1:
        print(f"Duplicate Config File Error. >> version : {version}")
        raise AssertionError

    with open(config_file_list[0], "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict


if __name__ == "__main__":
    inference()

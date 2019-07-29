from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch

from util.command_option import get_version
from util.log_module import create_train_logger, get_train_logger


class Process(metaclass=ABCMeta):

    def __init__(self, fold):
        """
        process abs class.

        Notes
        -----
        [version, fold] is can't access from child class.
        So, log utils should be implemented in this class.
        """
        self.__version = str(get_version())
        self.__fold = str(fold)

        log_list = ["epoch", "train_loss", "valid_loss", "train_qwk", "valid_qwk"]
        self.__log_df = pd.DataFrame(index=None, columns=log_list)
        create_train_logger(self.__version + "_" + self.__fold)
        get_train_logger(self.__version + "_" + self.__fold).debug("\t".join(log_list))

    @abstractmethod
    def data_preprocess():
        """
        Preprocess image and make dataloader instance.
        """
        raise NotImplementedError

    @abstractmethod
    def load_condition():
        """
        Prepare training condition and some module.(ex. model, optimizer, metric, ...etc
        """
        raise NotImplementedError

    @abstractmethod
    def training():
        """
        Training and calculation validation score.
        """
        raise NotImplementedError

    def update_best_model(self, model_weight):
        torch.save(model_weight, Path(__file__).parents[1] / "model" / "{}_{}.pth".format(self.__version, self.__fold))

    def update_training_log(self, log_dict):
        """
        Pytorch training log.
        """
        log_list = []
        for col in ["epoch", "train_loss", "valid_loss", "train_qwk", "valid_qwk"]:
            self.__log_df.loc[log_dict["epoch"], col] = log_dict[col]
            log_list.append(str(log_dict[col]))
        get_train_logger(self.__version + "_" + self.__fold).debug("\t".join(log_list))

    def update_learning_curve(self):
        save_path = Path(__file__).parents[1] / "log" / "figure"
        Path.mkdir(save_path, exist_ok=True, parents=True)

        # loss
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(self.__log_df["epoch"], self.__log_df["train_loss"], label="train")
        ax.plot(self.__log_df["epoch"], self.__log_df["valid_loss"], label="valid")
        plt.title("loss")
        plt.legend()
        plt.savefig(str(save_path / "loss.png"))

        # qwk
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.plot(self.__log_df["epoch"], self.__log_df["train_qwk"], label="train")
        ax.plot(self.__log_df["epoch"], self.__log_df["valid_qwk"], label="valid")
        plt.title("qwk")
        plt.legend()
        plt.savefig(str(save_path / "loss.png"))

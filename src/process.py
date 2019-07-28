from abc import ABCMeta, abstractmethod

from util.command_option import get_version


class Process(metaclass=ABCMeta):

    def __init__(self, fold):
        """
        process abs class.

        Notes
        -----
        [version, fold] is can't access from child class.
        So, log utils should be implemented in this class.
        """
        self.__version = get_version()
        self.__fold = fold

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

    def update_learning_curve(self):
        """
        Update Learning Curve figure.
        """
        raise NotImplementedError  # TODO

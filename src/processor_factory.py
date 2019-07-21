import gc
import yaml

from logging import getLogger
from pathlib import Path


class ProcessorFactory():
    """
    Abstract Factory Pattern
    Return experiment process by fold.(Each process return only one model.)

    Attributes
    ----------
    ROOT_PATH : PosixPath object
        project root path.
    """

    ROOT_PATH = Path(__file__).absolute().parents[1]

    @classmethod
    def make_process(cls, version):
        """
        Loading config and making process instance.

        Returns
        -------
        process : process instance
            training process instance.
        """
        pass

    @classmethod
    def __load_config(cls, version):
        """
        Loading yaml file.

        Parameters
        ----------
        version : str
            experiment version unique id.

        Returns
        -------
        config : dict
            information of process condition.
        """
        pass


if __name__ == "__main__":
    """
    unit test
    """
    pass

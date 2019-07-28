import yaml

from logging import getLogger
from pathlib import Path

from classifier_process import ClassifierProcess
from util.command_option import get_version


class ProcessorFactory():
    """
    Abstract Factory Pattern
    Return experiment process by fold.(Each process return only one model.)
    """

    @classmethod
    def make_process(self, fold):
        """
        Loading config and making process instance.

        Returns
        -------
        process : process instance
            training process instance.
        """
        config = self.__load_config()

        process = None

        if config["task"] == "classifier":
            process = ClassifierProcess(config)

        return process

    @classmethod
    def __load_config(self):
        """
        Loading yaml file.

        Returns
        -------
        config : dict
            information of process condition.
        """
        version = get_version()
        yaml_filepath = Path(__file__).parents[1] / "config" / f"{version}.yml"

        with open(yaml_filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        return config_dict


if __name__ == "__main__":
    """
    unit test
    """
    from process import Process
    assert isinstance(ProcessorFactory.make_process(0), Process)

    p = ProcessorFactory.make_process(0)
    p.data_preprocess()
    p.load_condition()
    p.training()

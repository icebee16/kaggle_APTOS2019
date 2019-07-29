import yaml

from pathlib import Path

from classifier_process import ClassifierProcess
from util.command_option import get_version


class ProcessorFactory():
    """
    Abstract Factory Pattern
    Return experiment process by fold.(Each process return only one model.)
    """

    @classmethod
    def make_process(self):
        """
        Loading config and making process instance.

        Returns
        -------
        process : process instance
            training process instance.
        """
        config = self.__load_config()

        process_list = []

        if config["summary"]["task"] == "classifier":
            for i in range(config["summary"]["fold"]):
                process_list.append(ClassifierProcess(config, i))
        elif config["summary"]["task"] == "regression":
            raise NotImplementedError

        return process_list

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
    p = ProcessorFactory.make_process()[0]
    p.data_preprocess()
    p.load_condition()
    p.training()
    # assert isinstance(p, Process)

import yaml
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from util.command_option import get_version
from util.kaggle_util import is_kagglekernel
from dataloader import custom_transforms


class executer(object):
    """
    exec inference
    """

    def __init__(self, fold):
        self.fold = fold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = self.__load_config()
        self.model = None
        self.test_loader = None
        self.predict = None

    def __input_path(self):
        img_dir = "input/aptos2019-blindness-detection" if is_kagglekernel() else "input"
        return Path(__file__).absolute().parents[1] / img_dir

    def __load_config(self):
        """
        Loading yaml file.

        Returns
        -------
        config : dict
            information of process condition.
        """
        version = get_version()
        config_dir = Path(__file__).parents[1] / "config"  # TODO
        config_file_list = list(config_dir.glob(f"{version}*.yml"))

        if len(config_file_list) > 1:
            print(f"Duplicate Config File Error. >> version : {version}")
            raise AssertionError

        with open(config_file_list[0], "r") as f:
            config_dict = yaml.safe_load(f)

        return config_dict

    def data_preprocess(self):
        """
        Make test dataloader.

        Notes
        -----
        Should implement "preprocess" part for dataset.

        dataloader reproducibility
        https://qiita.com/yagays/items/d413787a78aae825dbd3
        """
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        transform_config = self.config["dataloader"]["transform"]
        test_img_df = pd.read_csv(self.__input_path() / "test.csv")

        dataset_name = self.config["dataloader"]["dataset"]

        test_transform = self.__load_transforms(transform_config["test"])
        test_dataset_params = {"img_df": test_img_df, "transform": test_transform}
        test_dataset = self.__load_dataset(dataset_name, test_dataset_params)
        self.predict = np.zeros((len(test_dataset)))  # only test process
        self.test_loader = DataLoader(test_dataset,
                                      shuffle=True,
                                      num_workers=4,
                                      batch_size=self.config["dataloader"]["batch_size"],
                                      worker_init_fn=worker_init_fn)

    def __load_transforms(self, transforms_config):
        """
        Return transform function.

        Returns
        -------
        transform

        Notes
        -----
        Probabilistic behavior is NotImplemented.
        """

        transforms_config = [tf if "params" in tf.keys() else {"function": tf["function"], "params": {}} for tf in transforms_config]

        transform = []
        for tf in transforms_config:
            if hasattr(custom_transforms, tf["function"]):
                transform.append(getattr(custom_transforms, tf["function"])(**tf["params"]))
            elif hasattr(transforms, tf["function"]):
                transform.append(getattr(transforms, tf["function"])(**tf["params"]))
            else:
                print("Not Difine: {}".format(tf["function"]))
                raise AttributeError

        return transforms.Compose(transform)

    def __load_dataset(self, dataset_name, dataset_params):
        """
        loading dataset for test dataset.
        """
        module_name = "dataloader." + dataset_name.lower() + "_dataset"
        module = import_module(module_name)

        class_name = dataset_name.capitalize() + "TestDataset"
        dataset = getattr(module, class_name)(**dataset_params)

        return dataset

    def load_model(self):
        task_name = self.config["summary"]["task"]
        model_dir = Path(__file__).absolute().parents[1] / ("input/model" if is_kagglekernel() else "model")
        w_path = model_dir / "{}_{}.pth".format(get_version(), self.fold)
        model_config = self.config["train"]["model"]
        model_module = import_module("model." + model_config["name"])
        model = getattr(model_module, "get_" + model_config["name"])(task=task_name, weight=torch.load(w_path))
        self.model = model.to(self.device)

    def inference(self):
        batch_size = self.config["dataloader"]["batch_size"]
        self.model = self.model.to(self.device)
        self.model.eval()

        for i, data in enumerate(self.test_loader):
            inputs = data["image"].to(self.device, dtype=torch.float)
            outputs = self.model(inputs)
            if torch.cuda.is_available():
                outputs = outputs.cpu()
            self.predict[i * batch_size:(i + 1) * batch_size] = np.argmax(outputs.detach().numpy(), axis=1)

        submit_path = str(Path(__file__).absolute().parents[1] / "data" / "submit" / "{}.csv".format(get_version()))
        if is_kagglekernel():
            submit_path = "submission.csv"

        submission_df = pd.read_csv(self.__input_path() / "sample_submission.csv")
        submission_df["diagnosis"] = self.predict.astype(int)
        submission_df.to_csv(submit_path, index=False)


if __name__ == "__main__":
    e = executer(0)
    e.data_preprocess()
    e.load_model()
    e.inference()

import os
import random
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from process import Process
from dataloader import custom_transforms


class ClassifierProcess(Process):
    """
    training process for classifier.
    """

    def __init__(self, config, fold=0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        super().__init__(fold)
        self.config = config

        self.train_loader = None
        self.valid_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.__set_seed()

    def __set_seed(self):
        """
        Set seed for all module.
        """
        seed = self.config["train"]["condition"]["seed"]

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def data_preprocess(self):
        """
        Make train and valid dataloader.

        Notes
        -----
        Should implement "preprocess" part for dataset.

        dataloader reproducibility
        https://qiita.com/yagays/items/d413787a78aae825dbd3
        """
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        transform_config = self.config["dataloader"]["transform"]
        train_df = pd.read_csv(Path(__file__).parents[1] / "input" / "train.csv")
        train_img_df, valid_img_df = train_test_split(train_df,
                                                      test_size=0.2,
                                                      stratify=train_df["diagnosis"],
                                                      random_state=self.config["train"]["condition"]["seed"])
        train_img_df.reset_index(inplace=True)
        valid_img_df.reset_index(inplace=True)
        dataset_name = self.config["dataloader"]["dataset"]

        train_transform = self.__load_transforms(transform_config["train"])
        train_dataset_params = {"img_df": train_img_df, "transform": train_transform}
        train_dataset = self.__load_dataset(dataset_name, train_dataset_params)
        self.train_loader = DataLoader(train_dataset,
                                       shuffle=True,
                                       num_workers=4,
                                       batch_size=self.config["dataloader"]["batch_size"],
                                       worker_init_fn=worker_init_fn)

        valid_transform = self.__load_transforms(transform_config["valid"])
        valid_dataset_params = {"img_df": valid_img_df, "transform": valid_transform}
        valid_dataset = self.__load_dataset(dataset_name, valid_dataset_params)
        self.valid_loader = DataLoader(valid_dataset,
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
        loading dataset for train and valid dataset.
        """
        module_name = "dataloader." + dataset_name.lower() + "_dataset"
        module = import_module(module_name)

        class_name = dataset_name.capitalize() + "TrainDataset"
        dataset = getattr(module, class_name)(**dataset_params)

        return dataset

    def load_condition(self):
        # model
        model_config = self.config["train"]["model"]
        model_module = import_module("model." + model_config["name"])
        model = getattr(model_module, "get_" + model_config["name"])(task="classifier", pretrained=model_config["pretrained"])
        self.model = model.to(self.device)

        # optimizer
        optimizer_config = self.config["train"]["optimizer"]
        optimizer_config["params"]["params"] = self.model.parameters()
        self.optimizer = getattr(optim, optimizer_config["algorithm"])(**optimizer_config["params"])

        # scheduler
        scheduler_config = self.config["train"]["scheduler"]
        scheduler_config["params"]["optimizer"] = self.optimizer
        self.scheduler = getattr(lr_scheduler, scheduler_config["algorithm"])(**scheduler_config["params"])

        # criterion
        criterion_cofig = self.config["train"]["criterion"]
        self.criterion = getattr(nn, criterion_cofig["algorithm"])()

    def training(self):
        pass

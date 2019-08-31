import os
import random
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from fastprogress import master_bar, progress_bar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from process import Process
from dataloader import custom_transforms
from estimator import custom_loss
from estimator.optimized_qwk import OptimizedQWK
from util.log_module import stop_watch


class RegressionProcess(Process):
    """
    training process for regression
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
        self.estimator = None

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

    @stop_watch("data_preprocess()")
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

        if "eval_batch_size" not in self.config["dataloader"].keys():
            self.config["dataloader"]["eval_batch_size"] = self.config["dataloader"]["batch_size"]

        print(self.config["dataloader"]["eval_batch_size"])
        valid_transform = self.__load_transforms(transform_config["valid"])
        valid_dataset_params = {"img_df": valid_img_df, "transform": valid_transform}
        valid_dataset = self.__load_dataset(dataset_name, valid_dataset_params)
        self.valid_loader = DataLoader(valid_dataset,
                                       shuffle=True,
                                       num_workers=4,
                                       batch_size=self.config["dataloader"]["eval_batch_size"],
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
                raise NotImplementedError("Not Define: {}".format(tf["function"]))

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

    @stop_watch("load_condition()")
    def load_condition(self):
        # model
        model_config = self.config["train"]["model"]
        model_module = import_module("model." + model_config["name"])
        model = getattr(model_module, "get_" + model_config["name"])(task="regression", pretrained=model_config["pretrained"])
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
        if "params" not in criterion_cofig.keys():
            criterion_cofig["params"] = {}

        if hasattr(custom_loss, criterion_cofig["algorithm"]):
            self.criterion = getattr(custom_loss, criterion_cofig["algorithm"])(**criterion_cofig["params"])
        elif hasattr(nn, criterion_cofig["algorithm"]):
            self.criterion = getattr(nn, criterion_cofig["algorithm"])(**criterion_cofig["params"])
        else:
            raise NotImplementedError("Not Define: {}".format(criterion_cofig))

        # estimator
        self.estimator = OptimizedQWK()

    @stop_watch("training()")
    def training(self):
        condition = self.config["train"]["condition"]
        best_score = {"epoch": -1, "train_loss": np.inf, "valid_loss": np.inf, "train_qwk": 0.0, "valid_qwk": 0.0}

        non_improvement_round = 0
        mb = master_bar(range(condition["epoch"]))
        for epoch in mb:

            temp_score = {"epoch": epoch, "train_loss": 0.0, "valid_loss": 0.0, "train_qwk": 0.0, "valid_qwk": 0.0}
            for phase in ["train", "valid"]:
                if phase == "train":
                    data_loader = self.train_loader
                    self.scheduler.step()
                    self.model.train()
                    self.estimator.init_coef()
                elif phase == "valid":
                    data_loader = self.valid_loader
                    self.model.eval()

                running_loss = 0.0
                y_true, y_pred = np.array([]).reshape((0, 1)), np.array([]).reshape((0, 1))
                for data in progress_bar(data_loader, parent=mb):
                    mb.child.comment = ">> {} phase".format(phase)
                    inputs = data["image"].to(self.device, dtype=torch.float)
                    labels = data["label"].view(-1, 1).to(self.device, dtype=torch.float)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    with torch.set_grad_enabled(phase == "train"):
                        loss = self.criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item()

                    if torch.cuda.is_available():
                        labels = labels.cpu()
                        outputs = outputs.cpu()
                    y_true = np.vstack((y_true, labels.detach().numpy()))
                    y_pred = np.vstack((y_pred, outputs.detach().numpy()))

                if phase == "train":
                    self.estimator.fit(y_true, y_pred)
                y_pred = self.estimator.discretization(y_pred)
                temp_score["{}_loss".format(phase)] = running_loss / len(data_loader)
                temp_score["{}_qwk".format(phase)] = self.__qwk_scoring(y_true, y_pred)

            super().update_training_log(temp_score)

            if best_score["valid_loss"] > temp_score["valid_loss"]:
                best_score = temp_score
                super().update_best_model(self.model.state_dict())
                super().update_best_qwk_coef(self.estimator.get_coef())
                non_improvement_round = 0
            else:
                non_improvement_round += 1

            if epoch % 10 == 0:
                text = "[epoch {}] best epoch:{}  train loss:{}  valid loss:{}  train qwk:{}  valid qwk:{}".format(
                    epoch,
                    best_score["epoch"],
                    np.round(best_score["train_loss"], 5),
                    np.round(best_score["valid_loss"], 5),
                    np.round(best_score["train_qwk"], 5),
                    np.round(best_score["valid_qwk"], 5)
                )
                mb.write(text)
                super().update_learning_curve()

            # Early Stopping
            if non_improvement_round >= condition["early_stopping_rounds"]:
                print("\t Early stopping: {}[epoch]".format(epoch))
                break

        super().update_learning_curve()
        return best_score

    def __qwk_scoring(self, y_true, y_pred):
        score = cohen_kappa_score(y_true.reshape(-1),
                                  y_pred.reshape(-1),
                                  labels=[0, 1, 2, 3, 4],
                                  weights="quadratic")
        return score

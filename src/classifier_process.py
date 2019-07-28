import os
import random
import numpy as np
import torch
from torchvision import transforms

from process import Process
from dataloader import custum_transforms


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

        Notes
        -----
        reproducibility
        https://qiita.com/yagays/items/d413787a78aae825dbd3
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
        """
        tranform_config = self.config["dataloader"]["transform"]
        train_transform = self.__load_transforms(tranform_config["train"])
        print(train_transform)
        pass

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
            if hasattr(custum_transforms, tf["function"]):
                transform.append(getattr(custum_transforms, tf["function"])(**tf["params"]))
            elif hasattr(transforms, tf["function"]):
                transform.append(getattr(transforms, tf["function"])(**tf["params"]))
            else:
                print("Not Difine: {}".format(tf["function"]))
                raise AttributeError

        return transforms.Compose(transform)

    def load_condition(self):
        pass

    def training(self):
        pass

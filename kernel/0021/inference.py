import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from imgaug import augmenters as iaa
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models


# function
def get_version():
    return "0021"


def is_kagglekernel():
    return os.environ["HOME"] == "/tmp"
    # return True


# transforms
class custom_transforms():

    class WhiteNoise(object):
        """
        apply white noise


        """

        def __init__(self, prob, scale):
            self.prob = prob
            self.seq = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale)
            ])

        def __call__(self, img):
            rand = np.random.rand()
            if rand < self.prob:
                arr = np.array(img)
                arr = self.seq.augment_image(arr)
                img = Image.fromarray(arr)
            return img

    class RandomEraser(object):
        """
        """
        def __init__(self, prob, size_range=(0.02, 0.4), ratio_range=(0.3, 3)):
            self.prob = prob
            self.size_range = size_range  # TODO error case
            self.ratio_range = ratio_range

        def __call__(self, img):
            rand = np.random.rand()
            if rand < self.prob:
                arr = np.array(img)
                mask_value = np.random.randint(0, 256)

                h, w, _ = arr.shape

                mask_area = np.random.randint(h * w * self.size_range[0], h * w * self.size_range[1])

                mask_aspect_ratio = np.random.rand() * (self.ratio_range[1] - self.ratio_range[0]) + self.ratio_range[0]

                mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
                if mask_height > h - 1:
                    mask_height = h - 1
                mask_width = int(mask_aspect_ratio * mask_height)
                if mask_width > w - 1:
                    mask_width = w - 1

                top = np.random.randint(0, h - mask_height)
                left = np.random.randint(0, w - mask_width)
                bottom = top + mask_height
                right = left + mask_width

                arr[top:bottom, left:right].fill(mask_value)
                img = Image.fromarray(arr)
            return img


# dataset
class CircleTestDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        if is_kagglekernel():
            self.data_path = "../input/aptos2019-blindness-detection/test_images"
        else:
            self.data_path = Path(__file__).absolute().parents[2] / "input" / "test_images"

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        data_filepath = self.data_path + "/{}.png".format(self.img_df.loc[idx, "id_code"])
        img = cv2.imread(data_filepath)
        cropper = EdgeCrop()
        img = cropper(img)
        img = Image.fromarray(img)
        img = self.transform(img)

        return {"image": img}


class EdgeCrop(object):
    def __init__(self, center_search_loop=5000):
        self.loop = center_search_loop

    def _edge_detection(self, img):
        dst = cv2.medianBlur(img, ksize=7)
        sub = cv2.addWeighted(dst, 4, cv2.GaussianBlur(dst, (0, 0) , 50), -1, 80)
        _b, _g, sub = cv2.split(sub)
        _b, _g, dst = cv2.split(dst)
        dst = cv2.addWeighted(dst, 0.5, sub, 0.5, 0)
        _, dst = cv2.threshold(dst, np.mean(dst) / 2, 255, cv2.THRESH_BINARY)
        dst = cv2.Canny(dst, 0, 100)
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

        _, dst = cv2.threshold(dst, 10, 255, cv2.THRESH_BINARY)
        return dst

    def _calc_center_circle(self, edge_img, loop=5000):
        def calc_center_pixcel(A, B, C, D):
            def calc_lineparams(ax, ay, bx, by):
                if (by - ay) == 0:
                    by = by + 1
                slope = (ax - bx) / (by - ay)
                section = ((by**2 - ay**2) - (ax**2 - bx**2)) / (2 * (by - ay))
                return slope, section

            A_slope, A_section = calc_lineparams(A[0], A[1], B[0], B[1])
            B_slope, B_section = calc_lineparams(C[0], C[1], D[0], D[1])

            if abs(A_slope - B_slope) < 0.01:
                return None, None

            X = (B_section - A_section) / (A_slope - B_slope)
            Y = (A_slope * X + A_section + B_slope * X + B_section) / 2

            return int(X), int(Y)

        edge_list = np.where(edge_img[:, :, 2] == 255)
        if len(edge_list[0]) == 0:
            (h, w, c) = edge_img.shape
            (X, Y) = (int(w / 2), int(h / 2))
            radius = int(h / 2)
            return (X, Y), radius

        edge_list = [(edge_list[1][i], edge_list[0][i]) for i in range(len(edge_list[0]))]
        X_cand, Y_cand = [], []
        for _ in range(loop):
            edge = []
            edge.extend(edge_list[i] for i in np.random.randint(0, int(len(edge_list) / 2), 2))
            edge.extend(edge_list[i] for i in np.random.randint(int(len(edge_list) / 2), len(edge_list), 2))
            x, y = calc_center_pixcel(edge[0], edge[2], edge[1], edge[3])
            if x is not None:
                X_cand.append(x)
                Y_cand.append(y)

        X, Y = int(np.mean(X_cand)), int(np.mean(Y_cand))
        r_list = [np.sqrt((X - e[0]) ** 2 + (Y - e[1])**2) for e in edge_list]
        radius = int(np.median(r_list))
        return (X, Y), radius

    def _center_crop(self, img, center, radius):
        height, width, _ = img.shape
        mask = np.zeros((height, width), np.uint8)

        mask = cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
        mask_img = cv2.bitwise_and(img, img, mask=mask)

        crop_img = np.zeros((radius * 2, radius * 2, 3), np.uint8)
        cl, cr, ct, cb = 0, radius * 2, 0, radius * 2
        il, ir, it, ib = 0, width, 0, height
        if center[1] - radius > 0:
            it = center[1] - radius
        else:
            ct = radius - center[1]

        if height - center[1] > radius:
            ib -= (height - center[1]) - radius
        else:
            cb -= radius - (height - center[1])

        if center[0] - radius > 0:
            il = center[0] - radius
        else:
            cl = radius - center[0]

        if width - center[0] > radius:
            ir -= (width - center[0]) - radius
        else:
            cr -= radius - (width - center[0])

        crop_img[ct:cb, cl:cr, :] = mask_img[it:ib, il:ir, :]
        return crop_img

    def __call__(self, img):
        edge = self._edge_detection(img)
        center, radius = self._calc_center_circle(edge, loop=self.loop)
        img = self._center_crop(img, center=center, radius=radius)
        return img


# model
def get_mobilenet_v2(task, weight=None, pretrained=False):
    """
    Parameters
    ----------
    task: str
        kind of task.
    weight: torch.dict
        pretrained weight in training section.
        so, this parameter use in inference.
    pretrained: bool
        load torchvision model weight.
    """
    model = models.mobilenet_v2(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(Path(__file__).parents[2] / "model" / "pretrain" / "mobilenet_v2.pth"))

    num_features = model.classifier[1].in_features
    if task == "classifier":
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_features, out_features=5),
        )
    elif task == "regression":
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_features, out_features=1)
        )
    else:
        print("{} task isn't implemented.".format(task))
        raise NotImplementedError

    if weight is not None:
        model.load_state_dict(weight)

    return model


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

    def __input_path(self):
        if is_kagglekernel():
            img_dir = "../input/aptos2019-blindness-detection"
        else:
            img_dir = "../../input"
        return img_dir

    def __load_config(self):
        config_dict = {'summary': {'task': 'classifier', 'fold': 1}, 'dataloader': {'dataset': 'circle', 'batch_size': 16, 'preprocess': None, 'transform': {'train': [{'function': 'RandomRotation', 'params': {'degrees': 180}}, {'function': 'RandomHorizontalFlip'}, {'function': 'Resize', 'params': {'size': 244}}, {'function': 'ToTensor'}, {'function': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}], 'valid': [{'function': 'RandomRotation', 'params': {'degrees': 180}}, {'function': 'RandomHorizontalFlip'}, {'function': 'Resize', 'params': {'size': 244}}, {'function': 'ToTensor'}, {'function': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}], 'test': [{'function': 'RandomRotation', 'params': {'degrees': 180}}, {'function': 'RandomHorizontalFlip'}, {'function': 'Resize', 'params': {'size': 244}}, {'function': 'ToTensor'}, {'function': 'Normalize', 'params': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}]}}, 'train': {'condition': {'epoch': 150, 'early_stopping_rounds': 10, 'verbose': 1, 'seed': 1116}, 'model': {'name': 'mobilenet_v2', 'pretrained': True}, 'optimizer': {'algorithm': 'Adam', 'params': {'lr': 0.001}}, 'scheduler': {'algorithm': 'StepLR', 'params': {'step_size': 5, 'gamma': 0.9}}, 'criterion': {'algorithm': 'CrossEntropyLoss'}}, 'inference': None}

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
        test_img_df = pd.read_csv(self.__input_path() + "/test.csv")

        dataset_name = self.config["dataloader"]["dataset"]

        test_transform = self.__load_transforms(transform_config["test"])
        test_dataset_params = {"img_df": test_img_df, "transform": test_transform}
        test_dataset = self.__load_dataset(dataset_name, test_dataset_params)
        self.predict = np.zeros((len(test_dataset)))  # only test process
        self.test_loader = DataLoader(test_dataset,
                                      shuffle=False,
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
        dataset = CircleTestDataset(**dataset_params)

        return dataset

    def load_model(self):
        task_name = self.config["summary"]["task"]
        model_dir = "../input/model{}_aptos2019".format(get_version())
        w_path = model_dir + "/{}_{}.pth".format(get_version(), self.fold)
        model = get_mobilenet_v2(task=task_name, weight=torch.load(w_path))
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

        submission_df = pd.read_csv(self.__input_path() + "/sample_submission.csv")
        submission_df["diagnosis"] = self.predict.astype(int)
        submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    e = executer(0)
    e.data_preprocess()
    e.load_model()
    e.inference()

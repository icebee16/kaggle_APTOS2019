from pathlib import Path

import numpy as np

from PIL import Image, ImageFile
import cv2

from torchvision import transforms
from torch.utils.data import Dataset

from util.kaggle_util import is_kagglekernel
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BencolorTrainDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        if is_kagglekernel():
            self.data_path = Path(__file__).parents[4] / "aptos2019-blindness-detection" / "train_images"
            self.cache_path = Path(__file__).parents[5] / "working" / "data" / "bencolor"
        else:
            self.data_path = Path(__file__).parents[2] / "input" / "train_images"
            self.cache_path = Path(__file__).parents[2] / "data" / "bencolor"

        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])

        if not cache_filepath.exists():
            self.__save_cache(idx)

        img = Image.open(str(cache_filepath))
        img = self.transform(img)

        label = self.img_df.loc[idx, "diagnosis"]

        return {"image": img, "label": label}

    def __save_cache(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = cv2.imread(str(img_filepath.resolve()))

        cropper = BenColorCrop()
        img = cropper(img)

        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        cv2.imwrite(str(cache_filepath), img)


class BencolorTestDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        if is_kagglekernel():
            self.data_path = Path(__file__).parents[4] / "aptos2019-blindness-detection" / "test_images"
            self.cache_path = Path(__file__).parents[5] / "working" / "data" / "bencolor"
        else:
            self.data_path = Path(__file__).parents[2] / "input" / "test_images"
            self.cache_path = Path(__file__).parents[2] / "data" / "bencolor"

        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])

        if not cache_filepath.exists():
            self.__save_cache(idx)

        img = Image.open(str(cache_filepath.resolve()))
        img = self.transform(img)

        return {"image": img}

    def __save_cache(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = cv2.imread(str(img_filepath.resolve()))

        cropper = BenColorCrop()
        img = cropper(img)

        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        cv2.imwrite(str(cache_filepath), img)


class BenColorCrop(object):
    def __init__(self, img_size=512, sigmaX=10, tol=7):
        self.img_size = img_size
        self.sigmaX = sigmaX
        self.tol = tol

    def __call__(self, img):
        img = self._load_ben_color(img)
        return img

    def _load_ben_color(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._crop_image_from_gray(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), self.sigmaX), -4, 128)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _crop_image_from_gray(self, img):
        if img.ndim == 2:
            mask = img > self.tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > self.tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

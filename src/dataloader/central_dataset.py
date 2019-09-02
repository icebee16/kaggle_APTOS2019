from pathlib import Path

import numpy as np

from PIL import Image, ImageFile
import cv2

from torchvision import transforms
from torch.utils.data import Dataset

from util.kaggle_util import is_kagglekernel

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CentralTrainDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor(), binary=False):
        self.img_df = img_df
        self.transform = transform
        self.binary = binary

        if is_kagglekernel():
            self.data_path = Path(__file__).parents[4] / "aptos2019-blindness-detection" / "train_images"
            self.cache_path = Path(__file__).parents[5] / "working" / "data" / "central"
        else:
            self.data_path = Path(__file__).parents[2] / "input" / "train_images"
            self.cache_path = Path(__file__).parents[2] / "data" / "central"

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
        if self.binary:
            label = 1 if label > 0 else 0

        return {"image": img, "label": label}

    def __save_cache(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = cv2.imread(str(img_filepath.resolve()))

        cropper = CentralCrop()
        img = cropper(img)

        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        cv2.imwrite(str(cache_filepath), img)


class CentralTestDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        if is_kagglekernel():
            self.data_path = Path(__file__).parents[4] / "aptos2019-blindness-detection" / "test_images"
            self.cache_path = Path(__file__).parents[5] / "working" / "data" / "central"
        else:
            self.data_path = Path(__file__).parents[2] / "input" / "test_images"
            self.cache_path = Path(__file__).parents[2] / "data" / "central"

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

        cropper = CentralCrop()
        img = cropper(img)

        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        cv2.imwrite(str(cache_filepath), img)


class CentralCrop(object):
    def __init__(self, tol=7, img_size=512):
        self.tol = tol
        self.img_size = img_size

    def __call__(self, img):
        img = self._circle_crop_v2(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
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

    def _circle_crop_v2(self, img):
        """
        Create circular crop around image centre
        """
        img = self._crop_image_from_gray(img)

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = self._crop_image_from_gray(img)

        return img

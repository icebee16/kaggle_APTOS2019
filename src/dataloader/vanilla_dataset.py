from pathlib import Path

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


from util.kaggle_util import is_kagglekernel


class VanillaTrainDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        if is_kagglekernel():
            self.data_path = Path(__file__).parents[4] / "aptos2019-blindness-detection" / "train_images"
        else:
            self.data_path = Path(__file__).parents[2] / "input" / "train_images"

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = Image.open(str(img_filepath.resolve()))
        img = self.transform(img)

        label = self.img_df.loc[idx, "diagnosis"]

        return {"image": img, "label": label}


class VanillaTestDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        if is_kagglekernel():
            self.data_path = Path(__file__).parents[4] / "aptos2019-blindness-detection" / "test_images"
        else:
            self.data_path = Path(__file__).parents[2] / "input" / "test_images"

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = Image.open(str(img_filepath.resolve()))
        img = self.transform(img)

        return {"image": img}

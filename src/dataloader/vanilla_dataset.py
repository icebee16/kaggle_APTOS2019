from pathlib import Path

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class VanillaTrainDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform
        self.data_path = Path(__file__).parents[2] / "input" / "train_images"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = Image.open(str(img_filepath.resolve()))
        img = self.transform(img)

        label = self.df.loc[idx, "diagnosis"]

        return {"image": img, "label": label}


class VanillaTestDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform
        self.data_path = Path(__file__).parents[2] / "input" / "test_images"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = Image.open(str(img_filepath.resolve()))
        img = self.transform(img)

        return {"image": img}

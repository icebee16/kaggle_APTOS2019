# Edge Crop
# copy from circle_dataset.py and accelerate

from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageFile

from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BenedgeTrainDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        self.data_path = Path(__file__).parents[2] / "input" / "train_images"
        self.cache_path = Path(__file__).parents[2] / "data" / "benedge"

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

        cropper = BenedgeCrop()
        img = cropper(img)

        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        cv2.imwrite(str(cache_filepath), img)


class BenedgeTestDataset(Dataset):
    def __init__(self, img_df, transform=transforms.ToTensor()):
        self.img_df = img_df
        self.transform = transform

        self.data_path = Path(__file__).parents[2] / "input" / "test_images"
        self.cache_path = Path(__file__).parents[2] / "data" / "benedge"

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

        return {"image": img}

    def __save_cache(self, idx):
        img_filepath = self.data_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        img = cv2.imread(str(img_filepath.resolve()))

        cropper = BenedgeCrop()
        img = cropper(img)

        cache_filepath = self.cache_path / "{}.png".format(self.img_df.loc[idx, "id_code"])
        cv2.imwrite(str(cache_filepath), img)


class BenedgeCrop(object):
    """
    from EdgeCrop
    diff: __call__()
    """
    def __init__(self, center_search_loop=5000, img_size=512):
        self.loop = center_search_loop
        self.img_size = img_size

    def _edge_detection(self, img):
        dst = cv2.medianBlur(img, ksize=5)
        sub = cv2.addWeighted(dst, 4, cv2.GaussianBlur(dst, (7, 7) , 50), -1, 80)
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
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
        return img


if __name__ == "__main__":
    file_list = ["01c31b10ab99.png", "0229c0a80d42.png"]
    img_dir = Path(__file__).absolute().parents[2] / "input" / "test_images"

    def __crop(img):
        cropper = BenedgeCrop()
        img = cropper(img)
        print(img.shape)

    for f in file_list:
        print("true")
        img = cv2.imread(str(img_dir / f))
        __crop(img)

        print("zero p")
        z_img = np.zeros_like(img)
        __crop(z_img)

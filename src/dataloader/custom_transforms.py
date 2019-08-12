import numpy as np

from PIL import Image
from imgaug import augmenters as iaa


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
            mask_value = np.random.randing(0, 256)

            h, w, _ = arr.shape

            mask_area = np.random.randint(h * w * self.size_range[0], h * w * self.s[1])

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
            img = Image.fromarry(arr)
        return img

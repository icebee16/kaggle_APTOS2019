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
        if rand > self.prob:
            arr = np.array(img)
            arr = self.seq.augment_image(arr)
            img = Image.fromarray(arr)
        return img

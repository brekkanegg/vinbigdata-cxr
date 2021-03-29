import cv2
import numpy as np
import math

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def train_multi_augment12(image):
    h, w = image.shape[0], image.shape[1]

    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_REPLICATE,
                p=1,
            ),
            A.RandomSizedCrop(
                min_max_height=(int(h * 0.9), h), height=h, width=w, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ],
        p=1,
    )
    augmented = aug(image=image)

    return augmented

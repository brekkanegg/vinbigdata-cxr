import cv2
import numpy as np
import math

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    CLAHE,
    HueSaturationValue,
    RandomBrightnessContrast,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    OneOf,
    ToFloat,
    ShiftScaleRotate,
    GridDistortion,
    ElasticTransform,
    JpegCompression,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    Blur,
    MotionBlur,
    MedianBlur,
    GaussNoise,
    CenterCrop,
    IAAAdditiveGaussianNoise,
    GaussianBlur,
    OpticalDistortion,
    RandomSizedCrop,
    BboxParams,
)


def train_multi_augment12(image, bboxes=None, category_id=None):
    h, w = image.shape[0], image.shape[1]
    if bboxes is not None:
        aug = Compose(
            [
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1,
                ),
                RandomSizedCrop(
                    min_max_height=(int(h * 0.9), h), height=h, width=w, p=0.25
                ),
                RandomBrightnessContrast(
                    brightness_limit=0.0, contrast_limit=0.3, p=0.25
                ),
            ],
            p=1,
            bbox_params=BboxParams(format="pascal_voc", label_fields=["category_id"]),
        )
        augmented = aug(image=image, bboxes=bboxes, category_id=category_id)

    else:  # Normal
        aug = Compose(
            [
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=1,
                ),
                RandomSizedCrop(
                    min_max_height=(int(h * 0.9), h), height=h, width=w, p=0.25
                ),
                RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.25
                ),
            ],
            p=1,
        )
        augmented = aug(image=image)

    return augmented

from .vin import *

from .augmentations import *


def get_dataset(cfgs, mode, specific_csv=None):
    if mode == "train":
        transform = get_augmentation(cfgs)
    else:
        transform = None

    dataset = VIN(cfgs, transform=transform, mode=mode)

    return dataset


def get_augmentation(cfgs):
    if cfgs["meta"]["inputs"]["augment"] == "train_multi_augment12":
        aug_fn = train_multi_augment12
    elif cfgs["meta"]["inputs"]["augment"]["augment"] is None:
        aug_fn = None

    return aug_fn


def get_collater(mode="train"):
    if mode == "train":
        collate_fn = collater
    else:
        collate_fn = collater_test

    return collate_fn

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
    if cfgs["model"]["inputs"]["augment"] == "train_multi_augment12":
        aug_fn = train_multi_augment12
    elif cfgs["inputs"]["augment"]["augment"] is None:
        aug_fn = None

    return aug_fn


def get_collater():
    collate_fn = collater

    return collate_fn
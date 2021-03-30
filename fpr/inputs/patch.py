# META_CSV

import os, sys
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import cv2
import random
import torch
from torch.utils.data import Dataset
from scipy import ndimage
import time
import pickle
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader
import json
from sklearn.model_selection import train_test_split

from . import augmentations


FINDINGS = [
    "Aortic enlargement",  ### 0 614
    "Atelectasis",  ### 1 37
    "Calcification",  ### 2 90
    "Cardiomegaly",  ### 3 460
    "Consolidation",  ### 4 71
    "ILD",  ### 5 77
    "Infiltration",  ### 6 123
    "Lung Opacity",  ### 7 264
    "Nodule/Mass",  ### 8 165
    "Other lesion",  ### 9 227
    "Pleural effusion",  ### 10 206
    "Pleural thickening",  ### 11 396
    "Pneumothorax",  ### 12 20
    "Pulmonary fibrosis",  ### 13 323
    "No finding",  ### 14 2121
]


def get_dataloader(cfgs, mode="train"):

    if mode == "train":
        transform = augmentations.train_multi_augment12
        collate_fn = collater

    elif mode == "val":
        transform = None
        collate_fn = collater

    elif mode == "test":

        transform = None
        collate_fn = collater_test

    _dataset = PATCH(cfgs, transform=transform, mode=mode)

    _loader = DataLoader(
        dataset=_dataset,
        batch_size=cfgs["batch_size"],
        num_workers=cfgs["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        sampler=None,
    )

    return _loader


def collater(data):

    fps = [s["fp"] for s in data]
    imgs = torch.tensor([s["img"] for s in data])
    labels = torch.tensor([s["label"] for s in data])

    return {"fp": fps, "img": imgs, "label": labels}


def collater_test(data):

    fps = [s["fp"] for s in data]
    imgs = torch.tensor([s["img"] for s in data])

    return {"fp": fps, "img": imgs}


class PATCH(Dataset):
    def __init__(self, cfgs, transform=None, mode="train"):

        self.cfgs = cfgs
        self.inputs_cfgs = cfgs["meta"]["inputs"]
        self.data_dir = cfgs["data_dir"]
        self.mode = mode
        self.transform = transform
        self.label = self.cfgs["meta"]["inputs"]["label"]

        try:
            with open(
                self.data_dir + f"/patch_256/{self.label}/fold_dict.pickle", "rb"
            ) as f:
                fold_dict = pickle.load(f)
        except Exception as e:
            print(e)
            fps = glob.glob(self.data_dir + f"/patch_256/{self.label}/*_fp_*.png")
            gts = glob.glob(self.data_dir + f"/patch_256/{self.label}/*_gt_*.png")
            if len(fps) == 0 or len(gts) == 0:
                raise ("run make_dataset.pt first!")
            train_fps, val_fps = train_test_split(fps, test_size=0.1)
            train_gts, val_gts = train_test_split(gts, test_size=0.1)
            fold_dict = {
                "train_fps": train_fps,
                "val_fps": val_fps,
                "train_gts": train_gts,
                "val_gts": val_gts,
            }
            with open(
                self.data_dir + f"/patch_256/{self.label}/fold_dict.pickle", "wb"
            ) as f:
                pickle.dump(fold_dict, f)

        # random.shuffle(fps)
        # random.shuffle(gts)
        # train_fps, val_fps = fps[: int(len(fps) * 0.9)], fps[int(len(fps) * 0.9) :]
        # train_gts, val_gts = gts[: int(len(gts) * 0.9)], gts[int(len(gts) * 0.9) :]

        if self.mode == "train":
            self.fps = fold_dict["train_fps"]
            self.gts = fold_dict["train_gts"]
        elif self.mode == "val":
            self.fps = fold_dict["val_fps"]
            self.gts = fold_dict["val_gts"]
            self.tot_files = self.fps + self.gts

    def __len__(self):
        if self.mode == "train":
            samples_per_epoch = min(len(self.fps), len(self.gts)) * 2
        elif self.mode == "val":
            samples_per_epoch = len(self.tot_files)

        return samples_per_epoch

    def __getitem__(self, index):
        """
        Resize -> Windowing -> Augmentation
        """

        if self.mode == "train":
            if index % 2 == 0:
                file_path = self.gts[index // 2]
            else:
                file_path = self.fps[index // 2 + 1]
        else:
            file_path = self.tot_files[index]

        img = cv2.imread(file_path, -1)
        # img = img.astype(np.float32)

        ims = self.inputs_cfgs["image_size"]
        if img.shape[1] != self.inputs_cfgs["image_size"]:
            if self.data_dir.endswith("png_1024"):
                interpolation = cv2.INTER_LINEAR
            elif self.data_dir.endswith("png_1024l"):
                interpolation = cv2.INTER_LANCZOS4
            img = cv2.resize(img, (ims, ims), interpolation=interpolation)

        # FIXME: concat and windowing

        img = self.windowing(img)

        img = img.astype(np.float32)

        if self.cfgs["run"] == "test":
            data = {}
            data["fp"] = file_path
            data["img"] = img

        else:
            if "_gt_" in file_path:
                label = [1.0 - 0.001]
            elif "_fp_" in file_path:
                label = [0.0 + 0.001]

            if self.transform is not None:
                augmented = self.transform(img)
                img = augmented["image"]

            data = {}
            data["fp"] = file_path
            data["img"] = img
            data["label"] = label

        return data

    def windowing(self, img):
        if self.cfgs["meta"]["inputs"]["window"] == "cxr":

            eps = 1e-6
            center = img.mean()
            if self.mode == "train":
                width = img.std() * (random.random() + 3.5)
            else:
                width = img.std() * 4
            low = center - width / 2
            high = center + width / 2
            img = (img - low) / (high - low + eps)
            img[img < 0.0] = 0.0
            img[img > 1.0] = 1.0

            img = np.concatenate((img[:, :, np.newaxis],) * 3, axis=2)

        elif self.cfgs["meta"]["inputs"]["window"] == "imagenet":
            # if self.mode == "val":
            #     print("debug")

            img = (img - img.min()) / (img.max() - img.min())

            img = np.concatenate((img[:, :, np.newaxis],) * 3, axis=2)

            # do in augmentation
            # pass

            stat_mean = (0.485, 0.456, 0.406)
            stat_std = (0.229, 0.224, 0.225)

            img[:, :, 0] = (img[:, :, 0] - stat_mean[0]) / stat_std[0]
            img[:, :, 1] = (img[:, :, 1] - stat_mean[1]) / stat_std[1]
            img[:, :, 2] = (img[:, :, 2] - stat_mean[2]) / stat_std[2]

        else:
            raise

        return img


class InfiniteDataLoader(DataLoader):
    """Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
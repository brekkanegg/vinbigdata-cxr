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
        # transform = getattr(augmentations, cfgs["meta"]["inputs"]["augment"])
        transform = augmentations.train_multi_augment12
        collate_fn = collater

    elif mode == "val":
        # if cfgs["meta"]["inputs"]["augment"] == "imagenet":
        #     transform = augmentations.imagenet_val
        # else:
        transform = None
        collate_fn = collater

    elif mode == "test":
        # if cfgs["meta"]["inputs"]["augment"] == "imagenet":
        #     transform = augmentations.imagenet_val
        # else:
        transform = None
        collate_fn = collater_test

    _dataset = VIN(cfgs, transform=transform, mode=mode)

    _loader = DataLoader(
        dataset=_dataset,
        batch_size=cfgs["batch_size"],
        num_workers=cfgs["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        sampler=None,
    )

    # FIXME: shuffle 을 아이디 순서 바꾸는 걸로 하는데, infinitedataloader 쓰면 처음에 define 할 때 순서가 고정인 것 같아서 일단 변경
    # _loader = InfiniteDataLoader(
    #     dataset=_dataset,
    #     batch_size=cfgs["batch_size"],
    #     num_workers=cfgs["num_workers"],
    #     pin_memory=True,
    #     drop_last=False,
    #     collate_fn=collate_fn,
    #     sampler=None,
    # )

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


class VIN(Dataset):
    def __init__(self, cfgs, transform=None, mode="train"):

        self.cfgs = cfgs
        self.inputs_cfgs = cfgs["meta"]["inputs"]
        self.data_dir = cfgs["data_dir"]
        self.mode = mode
        self.transform = transform

        if self.mode != "test":
            with open(self.data_dir + "/train_meta_dict.pickle", "rb") as f:
                self.meta_dict = pickle.load(f)

            # self.fold_df = self.get_fold_df()

            self.pids = self.get_pids()

            if self.mode == "train":
                if self.cfgs["meta"]["train"]["posneg_ratio"] == 1:
                    self.abnormal_pids, self.normal_pids = self.split_abnormal_pids(
                        self.pids
                    )

        else:
            with open(self.data_dir + "/test_meta_dict.pickle", "rb") as f:
                self.meta_dict = pickle.load(f)
            self.pids = list(self.meta_dict.keys())

    def get_pids(self):
        with open(self.data_dir + "/sh_annot.json", "r") as f:
            annotations = json.load(f)
            # print(annotations.keys())

        if self.mode == "train":
            fold_list = [
                x for x in annotations["fold_indicator"] if x[-1] != self.cfgs["fold"]
            ]
            pids = np.array(fold_list)[:, 0].tolist()
        else:
            fold_list = [
                x for x in annotations["fold_indicator"] if x[-1] == self.cfgs["fold"]
            ]
            pids = np.array(fold_list)[:, 0].tolist()

        return pids

    def __len__(self):
        if self.cfgs["meta"]["train"]["samples_per_epoch"] is not None:
            samples_per_epoch = min(
                self.cfgs["meta"]["train"]["samples_per_epoch"], len(self.pids)
            )

        elif (self.mode == "train") and (
            self.cfgs["meta"]["train"]["posneg_ratio"] == 1
        ):
            samples_per_epoch = min(len(self.abnormal_pids), len(self.normal_pids)) * 2

        elif self.cfgs["meta"]["train"]["samples_per_epoch"] is None:
            samples_per_epoch = len(self.pids)

        return samples_per_epoch

    def __getitem__(self, index):
        """
        Resize -> Windowing -> Augmentation
        """

        pid = self.pids[index]
        if (self.mode == "train") and (self.cfgs["meta"]["train"]["posneg_ratio"] == 1):
            if index % 2 == 0:
                pid = self.abnormal_pids[index // 2]
            else:
                pid = self.normal_pids[index // 2 + 1]
        else:
            pid = self.pids[index]

        ims = self.inputs_cfgs["image_size"]

        if self.cfgs["run"] != "test":
            file_path = self.data_dir + f"/train/{pid}.png"
        else:
            file_path = self.data_dir + f"/test/{pid}.png"

        img = cv2.imread(file_path, -1)
        # img = img.astype(np.float32)

        if img.shape[1] != self.inputs_cfgs["image_size"]:
            if self.data_dir.endswith("png_1024"):
                interpolation = cv2.INTER_LINEAR
            elif self.data_dir.endswith("png_1024l"):
                interpolation = cv2.INTER_LANCZOS4
            img = cv2.resize(img, (ims, ims), interpolation=interpolation)

        # FIXME: concat and windowing

        img = self.windowing(img)

        # if self.cfgs["meta"]["inputs"]["window"] == "cxr":
        # img = np.concatenate((img[np.newaxis, :, :],) * 3, axis=0)
        # if self.cfgs["meta"]["inputs"]["window"] == "imagenet":

        # else:
        #     img = np.concatenate((img[:, :, np.newaxis],) * 3, axis=2)

        img = img.astype(np.float32)

        if self.cfgs["run"] == "test":
            data = {}
            data["fp"] = pid
            data["img"] = img

        else:
            pid_info = self.meta_dict[pid]

            pid_bbox = np.array(pid_info["bbox"])
            # pid_bbox order: rad_id, finding, finding_id, bbox(x_min, y_min, x_max, y_max) - xyxy가로, 세로
            pid_label = pid_bbox[:, 2]
            pid_rad = pid_bbox[:, 0]

            # CHECK if two not found exists
            # Normal
            if (pid_label == "14").all():
                label = [self.cfgs["meta"]["inputs"]["label_smooth"]]

            else:
                label = [1 - self.cfgs["meta"]["inputs"]["label_smooth"]]

            if self.transform is not None:
                augmented = self.transform(img)
                img = augmented["image"]

            data = {}
            data["fp"] = pid
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
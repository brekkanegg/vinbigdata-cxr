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

from . import nms


def collater(data):

    fps = [s["fp"] for s in data]
    imgs = torch.tensor([s["img"] for s in data])
    bboxes = [s["bbox"] for s in data]
    max_num_bboxes = max(bbox.shape[0] for bbox in bboxes)

    if max_num_bboxes > 0:
        bboxes_padded = torch.ones((len(bboxes), max_num_bboxes, 5)) * -1

        if max_num_bboxes > 0:
            for idx, bbox in enumerate(bboxes):
                if bbox.shape[0] > 0:
                    bboxes_padded[idx, : bbox.shape[0], :] = torch.tensor(bbox)
    else:
        bboxes_padded = torch.ones((len(bboxes), 1, 5)) * -1

    return {"fp": fps, "img": imgs, "bbox": bboxes_padded}


def collater_test(data):

    fps = [s["fp"] for s in data]
    imgs = torch.tensor([s["img"] for s in data])

    return {"fp": fps, "img": imgs}


class VIN(Dataset):
    def __init__(self, cfgs, transform=None, mode="train"):

        self.cfgs = cfgs
        self.inputs_cfgs = cfgs["model"]["inputs"]
        self.data_dir = cfgs["data_dir"]
        self.mode = mode
        self.transform = transform

        if self.mode != "test":
            with open(self.data_dir + "/train_meta_dict.pickle", "rb") as f:
                self.meta_dict = pickle.load(f)

            self.pids = self.get_train_pids()
            self.nms = getattr(nms, self.cfgs["model"]["inputs"]["nms"])
        else:
            with open(self.data_dir + "/test_meta_dict.pickle", "rb") as f:
                self.meta_dict = pickle.load(f)

            self.pids = list(self.meta_dict.keys())

    def get_train_pids(self):

        # FIXME: This is based on label distribution
        # MultilabelStratified KFold
        xs = np.array(list(self.meta_dict.keys()))
        ys = []
        for v in self.meta_dict.values():
            temp_lbl = np.array(v["bbox"])[:, 2]
            temp = np.zeros((self.cfgs["model"]["inputs"]["num_classes"]))
            for i in temp_lbl:
                temp[int(i)] = 1
            ys.append(temp)
        ys = np.array(ys)

        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=30)
        kfold_generator = mskf.split(xs, ys)

        for _ in range(self.cfgs["fold"] + 1):
            train_index, val_index = next(kfold_generator)

        if self.mode == "train":
            pids = xs[train_index]
        elif self.mode == "val":
            pids = xs[val_index]

        pids = [str(pid) for pid in pids]

        return pids

    def __len__(self):
        if self.cfgs["model"]["train"]["samples_per_epoch"] is None:
            samples_per_epoch = len(self.pids)
        else:
            samples_per_epoch = min(
                self.cfgs["model"]["train"]["samples_per_epoch"], len(self.pids)
            )
        return samples_per_epoch

    def __getitem__(self, index):
        """
        Resize -> Windowing -> Augmentation
        """

        pid = self.pids[index]
        ims = self.cfgs["model"]["inputs"]["image_size"]

        if self.cfgs["run"] != "test":
            file_path = self.data_dir + f"/train/{pid}.png"
        else:
            file_path = self.data_dir + f"/test/{pid}.png"

        img = cv2.imread(file_path, -1)
        img = img.astype(np.float32)

        if img.shape[1] != self.inputs_cfgs["image_size"]:
            img = cv2.resize(img, (ims, ims))

        img = self.windowing(img)
        img = img.astype(np.float32)

        img = np.concatenate((img[:, :, np.newaxis],) * 3, axis=2)

        if self.cfgs["run"] == "test":
            data = {}
            data["fp"] = pid
            data["img"] = img

        else:
            bboxes_coord = []
            bboxes_cat = []
            bboxes_rad = []

            pid_info = self.meta_dict[pid]
            pid_dimy = pid_info["dim0"]
            pid_dimx = pid_info["dim1"]

            pid_bbox = np.array(pid_info["bbox"])
            # pid_bbox order: rad_id, finding, finding_id, bbox(x_min, y_min, x_max, y_max) - xyxy가로, 세로
            pid_label = pid_bbox[:, 2]
            pid_rad = pid_bbox[:, 0]

            # CHECK if two not found exists
            # Normal
            if (pid_label == "14").all():
                bboxes = np.ones((1, 5)) * -1
                if self.transform is not None:
                    augmented = self.transform(img)
                    img = augmented["image"]

            else:
                for bi, bb in enumerate(pid_bbox):
                    bx0, by0, bx1, by1 = [float(i) for i in bb[-4:]]
                    blabel = int(bb[2])
                    brad = int(pid_rad[bi])

                    if blabel == 14:
                        continue

                    if (bx0 >= bx1) or (by0 >= by1):
                        continue
                    else:
                        # Resize
                        temp_bb = [None, None, None, None]
                        temp_bb[0] = np.round(bx0 / pid_dimx * ims)
                        temp_bb[1] = np.round(by0 / pid_dimy * ims)
                        temp_bb[2] = np.round(bx1 / pid_dimx * ims)
                        temp_bb[3] = np.round(by1 / pid_dimy * ims)

                        bboxes_coord.append(temp_bb)
                        bboxes_cat.append(blabel)
                        bboxes_rad.append(brad)

                # NOTE: Simple NMS for multi-labeler case
                if len(bboxes_coord) >= 2:  # ("cst" in mask_path[0]) and
                    bboxes_coord, bboxes_cat = self.nms(
                        bboxes_coord, bboxes_cat, bboxes_rad, iou_th=0.5
                    )

                img_anns = {
                    "image": img,
                    "bboxes": bboxes_coord,
                    "category_id": bboxes_cat,
                }

                if self.transform is not None:
                    img_anns = self.transform(**img_anns)

                img = img_anns["image"]
                bboxes = img_anns["bboxes"]
                cats = img_anns["category_id"]
                bboxes = [list(b) + [c] for (b, c) in zip(bboxes, cats)]
                bboxes = np.array(bboxes)

            data = {}
            data["fp"] = pid
            data["img"] = img
            data["bbox"] = bboxes

        return data

    def windowing(self, img):
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

        return img

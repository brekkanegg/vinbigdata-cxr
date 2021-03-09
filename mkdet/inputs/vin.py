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


class VIN(Dataset):
    def __init__(self, cfgs, transform=None, mode="train"):

        self.cfgs = cfgs
        self.inputs_cfgs = cfgs["model"]["inputs"]
        self.data_dir = cfgs["data_dir"]
        self.mode = mode
        self.transform = transform

        if (self.mode == "train") or (self.mode == "val"):
            with open(self.data_dir + "/train_meta_dict.pickle", "rb") as f:
                self.meta_dict = pickle.load(f)

        self.pids = self.get_pids()

        # TODO: val gt_df

    def get_pids(self):

        # FIXME: This is based on label distribution
        # MultilabelStratified KFold
        xs = np.array(list(self.meta_dict.keys()))
        ys = []
        for v in self.meta_dict.values():
            temp_lbl = np.array(v["bbox"])[:, 2]
            temp = np.zeros((15))
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
        t0 = time.time()

        pid = self.pids[index]

        # row = self.meta_df.loc[index]

        # FIXME:

        file_path = self.data_dir + f"/png_1024/{self.cfgs['run']}/{pid}.png"

        img = cv2.imread(file_path, -1)
        img = img.astype(np.float32)

        t1 = time.time()

        if img.shape[1] != self.inputs_cfgs["image_size"]:
            img = cv2.resize(
                img, (self.inputs_cfgs["image_size"], self.inputs_cfgs["image_size"])
            )

        img = self.windowing(img)
        img = img.astype(np.float32)

        t2 = time.time()

        # img = (img - img.min()) / (img.max() - img.min())
        img = np.concatenate((img[:, :, np.newaxis],) * 3, axis=2)

        bboxes_coord = []
        bboxes_cat = []

        pid_info = self.meta_dict[pid]
        pid_dimy = pid_info["dim0"]
        pid_dimx = pid_info["dim1"]

        pid_bbox = np.array(pid_info["bbox"])
        # pid_bbox order: rad_id, finding, finding_id, bbox(x_min, y_min, x_max, y_max) - xyxy가로, 세로
        pid_label = pid_bbox[:, 2]

        # CHECK if two not found exists
        # Normal
        if (pid_label == "14").all():
            bboxes = np.ones((1, 5)) * -1
            if self.transform is not None:
                augmented = self.transform(img)
                img = augmented["image"]

        else:
            # FIXME: if no fiding case do not add
            for bb in pid_bbox:
                bx0, by0, bx1, by1 = [float(i) for i in bb[-4:]]
                bl = int(bb[2])

                if bl == 14:
                    continue

                # FIXME:
                if (bx0 >= bx1) or (by0 >= by1):
                    continue
                else:
                    # Resize
                    temp_bb = [None, None, None, None]
                    temp_bb[0] = int(bx0 / pid_dimx * self.inputs_cfgs["image_size"])
                    temp_bb[1] = int(by0 / pid_dimy * self.inputs_cfgs["image_size"])
                    temp_bb[2] = int(bx1 / pid_dimx * self.inputs_cfgs["image_size"])
                    temp_bb[3] = int(by1 / pid_dimy * self.inputs_cfgs["image_size"])

                    if (np.array(temp_bb) > self.inputs_cfgs["image_size"]).any():
                        a = 1

                    bboxes_coord.append(temp_bb)
                    bboxes_cat.append(bl)

            # TODO:
            # FIXME:
            # NOTE: Simple NMS for multi-labeler case
            if len(bboxes_coord) >= 2:  # ("cst" in mask_path[0]) and
                bboxes_coord, bboxes_cat = simple_nms(bboxes_coord, bboxes_cat)

            t3 = time.time()

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

            t4 = time.time()

            if self.cfgs["do_profiling"]:
                print("\n*data read(abnormal)", t1 - t0)
                print("*data resize", t2 - t1)
                print("*data get mask", t3 - t2)
                print("*data transform", t4 - t3)

        data = {}
        data["fp"] = pid
        data["img"] = img
        data["bbox"] = bboxes

        return data

    # def shuffle(self):

    #     normal_meta_df = self.normal_meta_df
    #     if (not self.inputs_cfgs["posneg_ratio"] is None) and self.mode == "train":
    #         # if len(self.normal_meta_df) > len(self.abnormal_meta_df):
    #         normal_meta_df = normal_meta_df.sample(frac=1).reset_index(drop=True)
    #         r = self.inputs_cfgs["posneg_ratio"]
    #         normal_meta_df = normal_meta_df.iloc[: len(self.abnormal_meta_df) * r]

    #     meta_df = pd.concat((self.abnormal_meta_df, normal_meta_df), axis=0)
    #     meta_df.reset_index(inplace=True)

    #     self.meta_df = meta_df.sample(frac=1).reset_index(drop=True)

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


# Helper Functions


# def filter_bbox(bbox):

#     non_zero_coord = np.argwhere(mask != 0)

#     margin = 0
#     y0 = np.clip(np.min(non_zero_coord[:, 0]) - margin, 0, mask.shape[0])
#     y1 = np.clip(np.max(non_zero_coord[:, 0]) + margin, 0, mask.shape[0])
#     x0 = np.clip(np.min(non_zero_coord[:, 1]) - margin, 0, mask.shape[1])
#     x1 = np.clip(np.max(non_zero_coord[:, 1]) + margin, 0, mask.shape[1])

#     if (x0 <= x1) and (y0 <= y1):
#         # if ((x0 + 5) < x1) and ((y0 + 5) < y1):
#         return [x0, y0, x1, y1]


def calc_iou(bbox_a, bbox_b):
    """
    :param a: bbox list [min_y, min_x, max_y, max_x]
    :param b: bbox list [min_y, min_x, max_y, max_x]
    :return:
    """
    size_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    size_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    min_ab_y = max(bbox_a[0], bbox_b[0])
    min_ab_x = max(bbox_a[1], bbox_b[1])
    max_ab_y = min(bbox_a[2], bbox_b[2])
    max_ab_x = min(bbox_a[3], bbox_b[3])

    inter_ab = max(0, max_ab_y - min_ab_y) * max(0, max_ab_x - min_ab_x)

    return inter_ab / (size_a + size_b - inter_ab)


def check_overlap(bbox_a, bbox_b):
    """
    :param a: bbox list [min_y, min_x, max_y, max_x]
    :param b: bbox list [min_y, min_x, max_y, max_x]
    :return:
    """
    # Assume bbox_a should be bigger than bbox_b
    ov = 0
    if (
        (bbox_a[0] <= bbox_b[0])
        and (bbox_a[1] <= bbox_b[1])
        and (bbox_a[2] >= bbox_b[2])
        and (bbox_a[3] >= bbox_b[3])
    ):
        ov = 1

    return ov


# TODO:
def simple_nms(bboxes_coord, bboxes_cat, iou_th=0.4):
    bbox_sizes = np.array([(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in bboxes_coord])
    order = bbox_sizes.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            ov = check_overlap(bboxes_coord[order[i]], bboxes_coord[order[j]])
            if ov > iou_th:
                keep[order[j]] = False
            else:
                ov = calc_iou(bboxes_coord[order[i]], bboxes_coord[order[j]])
                if ov > iou_th:
                    keep[order[j]] = False

    bboxes_coord = [bb for (idx, bb) in enumerate(bboxes_coord) if keep[idx]]
    bboxes_cat = [bb for (idx, bb) in enumerate(bboxes_cat) if keep[idx]]

    return bboxes_coord, bboxes_cat
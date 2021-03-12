"""
Assume train_df(abnormal/normal), val_df(abnormal/normal) is prepared
For TOY setting use 161128_amc_abnormal, 161128_amc_normal
"""

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


DISEASE_NAME_DICT = {
    0: "00Normal",
    1: "01Nodule",
    2: "02Calcification",
    3: "03Consolidation",
    4: "04InterstitialOpacity",
    5: "05Bronchiectasis",
    6: "06Atelectasis",
    7: "07Fibrosis",
    8: "08MediastinalWidening",
    9: "09Cardiomegaly",
    10: "10PleuralEffusion",
    11: "11Pneumothorax",
    12: "12RibFracture",
    13: "13Consolidation-Pneumonia",
    14: "14Emphysema",
    15: "15Tuberculosis-active",
    16: "16Tuberculosis-inactive",
    17: "17Tuberculosis-indeterminate",
    18: "18Pneumoperitoneum",
    99: "99Etc",
}


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


class CXR(Dataset):
    def __init__(self, cfgs, transform=None, mode="train", specific_csv=None):

        self.cfgs = cfgs
        self.inputs_cfgs = cfgs["model"]["inputs"]
        self.data_dir = cfgs["data_dir"]
        self.mode = mode
        self.transform = transform

        self.mapping_label = self.get_mapping_label()
        self.csv_files = self.get_csv_files()

        self.ignore_label = [
            i for i in DISEASE_NAME_DICT.values() if not i in self.mapping_label.keys()
        ]
        # self.ignore_label.remove("00Normal")

        if specific_csv is None:
            self.abnormal_meta_df, self.normal_meta_df, self.meta_df = self.get_meta_df(
                self.csv_files
            )
        else:
            self.abnormal_meta_df, self.normal_meta_df, self.meta_df = self.get_meta_df(
                [specific_csv]
            )

    def get_mapping_label(self):
        if self.inputs_cfgs["label"] == "pneumoperitoneum":
            mapping = {
                "18Pneumoperitoneum": 1,
            }

        elif self.inputs_cfgs["label"] == "ribfracture":
            mapping = {
                "12RibFracture": 1,
            }

        else:
            raise NotImplementedError("Wrong label specified")

        return mapping

    def get_csv_files(self):
        if self.inputs_cfgs["label"] == "pneumoperitoneum":
            csv_files = [
                "png_1024/161128_amc_normal.csv",
                "png_1024/180301_amc_normal.csv",
                "png_1024/180405_amc_normal.csv",
                "png_1024/180116_snubh_normal.csv",
                "png_1024/201104_cstnew_normal.csv",
                "png_1024/201104_cstnew_abnormal.csv",
            ]
        elif self.inputs_cfgs["label"] == "ribfracture":
            csv_files = [
                "png_1024/161128_amc_abnormal.csv",
                "png_1024/201104_cst_abnormal.csv",
                "png_1024/201104_cstnew_abnormal.csv",
                "png_1024/210124_nipa_abnormal.csv",
                "png_1024/210124_rightfund_abnormal.csv",
                "png_1024/161128_amc_normal.csv",
                "png_1024/180301_amc_normal.csv",
                "png_1024/180405_amc_normal.csv",
                "png_1024/180116_snubh_normal.csv",
                "png_1024/201104_cst_normal.csv",
                "png_1024/201104_cstnew_normal.csv",
                "png_1024/210124_nipa_normal.csv",
                "png_1024/210124_rightfund_normal.csv",
            ]

        else:
            raise NotImplementedError("Wrong label specified")

        return csv_files

    def get_meta_df(self, csv_files):
        df_abnormal_lists = []
        df_normal_lists = []
        for csv_fp in csv_files:
            csv_dir = os.path.join(self.data_dir, csv_fp)
            df = pd.read_csv(csv_dir, index_col=[0])

            # FIXME: Assinging folds
            if (df["fold"] == -1).all():
                df.reset_index(inplace=True)
                df["fold"] = df.apply(lambda row: row.index % 5 + 1)

            if self.mode == "train":
                df = df[(df["fold"] != self.cfgs["fold"])]
            else:
                df = df[(df["fold"] == self.cfgs["fold"])]

            normal_conditions = [True] * len(df)

            for col in self.mapping_label.keys():
                if not col in df.columns:
                    continue
                normal_conditions = normal_conditions & (df[col] != 1)
            normal_df = df[normal_conditions]
            df_normal_lists.append(normal_df)

            abnormal_conditions = [False] * len(df)
            for col in self.mapping_label.keys():
                if not col in df.columns:
                    continue
                abnormal_conditions = abnormal_conditions | (df[col] == 1)
            abnormal_df = df[abnormal_conditions]
            df_abnormal_lists.append(abnormal_df)

        abnormal_meta_df = (
            pd.concat(df_abnormal_lists, sort=False).reset_index().drop("index", axis=1)
        )
        abnormal_meta_df["is_normal"] = 0.0

        normal_meta_df = (
            pd.concat(df_normal_lists, sort=False).reset_index().drop("index", axis=1)
        )
        normal_meta_df["is_normal"] = 1.0

        meta_df = pd.concat((abnormal_meta_df, normal_meta_df), axis=0)
        meta_df.reset_index(inplace=True)

        return abnormal_meta_df, normal_meta_df, meta_df

    def __getitem__(self, index):
        """
        Resize -> Windowing -> Augmentation
        """
        t0 = time.time()

        row = self.meta_df.loc[index]

        file_path = row["img_path"]
        if file_path.startswith("/nfs3/chestpa"):
            home_dir = self.data_dir.split("/png")[0]
            file_path = file_path.replace("/nfs3/chestpa", home_dir)

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

        if row["is_normal"] == 1:
            bboxes = np.ones((1, 5)) * -1
            if self.transform is not None:
                augmented = self.transform(img)
                img = augmented["image"]

            t3 = time.time()

            if self.cfgs["do_profiling"]:
                print("\n*data read(normal)", t1 - t0)
                print("*data resize", t2 - t1)
                print("*data transform", t3 - t2)

        else:
            # FIXME: 그때그때 bbox 뽑기
            mask_path = sorted(row["mask_path"].split(","))

            for mfp in mask_path:
                mlbl = mfp.split("_")[-3]  # FIXME: -3 이 변할 수도
                if mlbl in self.ignore_label:
                    continue

                m_chl = self.mapping_label[mlbl]

                temp = cv2.imread(mfp, -1) // 255
                if temp.shape[1] != self.inputs_cfgs["image_size"]:
                    temp = cv2.resize(
                        temp,
                        (img.shape[0], img.shape[1]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    if temp.max() == 0:  # Mask_Area has been reduced
                        continue

                # NOTE: Each mask contains only one bbox
                temp_bbox = extract_bbox(temp)

                if temp_bbox is not None:
                    temp_bbox = self.adjust_bbox_size(temp_bbox)
                    bboxes_coord.append(temp_bbox)
                    bboxes_cat.append(m_chl - 1)
                else:
                    continue

            if len(bboxes_coord) == 0:
                bboxes = np.ones((1, 5)) * -1
                if self.transform is not None:
                    augmented = self.transform(img)
                    img = augmented["image"]

                t3 = time.time()

            else:
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
        data["fp"] = file_path
        data["img"] = img
        data["bbox"] = bboxes

        return data

    def __len__(self):
        if self.cfgs["model"]["train"]["samples_per_epoch"] is None:
            samples_per_epoch = len(self.meta_df)
        else:
            samples_per_epoch = min(
                self.cfgs["model"]["train"]["samples_per_epoch"], len(self.meta_df)
            )

        return samples_per_epoch

    def shuffle(self):

        normal_meta_df = self.normal_meta_df
        if (not self.inputs_cfgs["posneg_ratio"] is None) and self.mode == "train":
            # if len(self.normal_meta_df) > len(self.abnormal_meta_df):
            normal_meta_df = normal_meta_df.sample(frac=1).reset_index(drop=True)
            r = self.inputs_cfgs["posneg_ratio"]
            normal_meta_df = normal_meta_df.iloc[: len(self.abnormal_meta_df) * r]

        meta_df = pd.concat((self.abnormal_meta_df, normal_meta_df), axis=0)
        meta_df.reset_index(inplace=True)

        self.meta_df = meta_df.sample(frac=1).reset_index(drop=True)

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

    def adjust_bbox_size(self, bbox):
        min_size = self.cfgs["model"]["inputs"]["min_size"]
        image_size = self.cfgs["model"]["inputs"]["image_size"]
        [x0, y0, x1, y1] = bbox
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        if x1 - x0 < min_size:
            rs = max(min_size, np.random.normal(2 * min_size, min_size / 2))
            x0 = max(0, cx - rs // 2)
            x1 = min(cx + rs // 2, image_size - 1)
        if y1 - y0 < min_size:
            rs = max(min_size, np.random.normal(2 * min_size, min_size / 2))
            y0 = max(0, cy - rs // 2)
            y1 = min(cy + rs // 2, image_size - 1)

        return [x0, y0, x1, y1]


# Helper Functions
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


def extract_bbox(mask):

    non_zero_coord = np.argwhere(mask != 0)

    margin = 0
    y0 = np.clip(np.min(non_zero_coord[:, 0]) - margin, 0, mask.shape[0])
    y1 = np.clip(np.max(non_zero_coord[:, 0]) + margin, 0, mask.shape[0])
    x0 = np.clip(np.min(non_zero_coord[:, 1]) - margin, 0, mask.shape[1])
    x1 = np.clip(np.max(non_zero_coord[:, 1]) + margin, 0, mask.shape[1])

    if (x0 <= x1) and (y0 <= y1):
        # if ((x0 + 5) < x1) and ((y0 + 5) < y1):
        return [x0, y0, x1, y1]


def simple_nms(bboxes_coord, bboxes_cat, iou_th=0.15):
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
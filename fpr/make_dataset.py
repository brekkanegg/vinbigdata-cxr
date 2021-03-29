import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pickle
import os, sys
from tqdm import tqdm
import glob
import argparse


# save gt patch
def make_gt_patch(train_df, finding):
    cat_df = train_df[train_df["class_id"] == finding[0]]
    pids = list(set(train_df["image_id"]))
    for pid in tqdm(pids):
        pid_cat_df = cat_df[cat_df["image_id"] == pid]
        if len(pid_cat_df) == 0:
            continue

        num = 0
        for i in range(len(pid_cat_df)):
            temp_row = pid_cat_df.iloc[i]
            x0 = temp_row["x_min"] / temp_row["width"] * 1024
            y0 = temp_row["y_min"] / temp_row["height"] * 1024
            x1 = temp_row["x_max"] / temp_row["width"] * 1024
            y1 = temp_row["y_max"] / temp_row["height"] * 1024
            cx = int(np.round((x0 + x1) / 2))
            cy = int(np.round((y0 + y1) / 2))

            img = cv2.imread(f"{DATA_DIR}/png_1024l/train/{pid}.png", -1)

            py0 = max(0, cy - 127)
            px0 = max(0, cx - 127)
            py1 = min(1024, cy + 128)
            px1 = min(1024, cx + 128)
            patch = img[py0:py1, px0:px1]

            if patch.shape != (256, 256):
                patch = cv2.resize(patch, (256, 256))

            patch_name = f"{DATA_DIR}/patch_256/{finding[1]}/{pid}_gt_{num}.png"
            is_saved = cv2.imwrite(patch_name, patch)
            if not is_saved:
                raise ("not saved!")
            num += 1


def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y2]

    """
    bboxes = bboxes.copy().astype(
        float
    )  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


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


def nms_naive(bboxes_coord, bboxes_score, iou_th=0.4):

    order = bboxes_score.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            ov = calc_iou(bboxes_coord[order[i]], bboxes_coord[order[j]])
            if ov > iou_th:
                keep[order[j]] = False
    keep = np.array(keep)

    bboxes_coord = bboxes_coord[keep]
    #     bboxes_score = bboxes_score(keep)
    #     bboxes = np.concatenate((bboxes_coord, bboxes_score[:, np.newaxis]), axis=1)

    return bboxes_coord


# save fp patch
# Get False Positives
def get_false_positives(train_df, finding):
    cat_df = train_df[train_df["class_id"] == finding[0]]
    tot_label_dirs = glob.glob(
        "/nfs3/minki/kaggle/vinbigdata-cxr/yolov5/runs/test/fold*_0326_0001/labels/*.txt"
    )

    false_positives = []
    for ldir in tqdm(tot_label_dirs):
        pid = ldir.split("labels/")[1][:-4]

        with open(ldir, "r") as f:
            temp = f.readlines()

        pred = np.genfromtxt(temp, delimiter=" ")
        try:
            pid_cat_pred = pred[pred[:, 0] == finding[0]]
        except IndexError:
            continue

        if len(pid_cat_pred) > 0:
            pid_cat_pred_xyxy = yolo2voc(1024, 1024, pid_cat_pred[:, 1:-1])
        else:
            continue

        pid_cat_score = pid_cat_pred[:, -1]

        if len(pid_cat_pred_xyxy) > 1:
            order = pid_cat_score.argsort()[::-1]
            keep = [True] * len(order)
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    ov = calc_iou(
                        pid_cat_pred_xyxy[order[i]], pid_cat_pred_xyxy[order[j]]
                    )
                    if ov > 0.1:
                        keep[order[j]] = False
            keep = np.array(keep)

            pid_cat_pred_xyxy = pid_cat_pred_xyxy[keep]

        pid_cat_gt = cat_df[(cat_df["image_id"] == pid)]
        # pid_cat_gt = pid_gt[(pid_gt["class_id"] == finding[0])]

        if len(pid_cat_gt) == 0:
            for picp in pid_cat_pred_xyxy:
                false_positives.append((pid, picp))
                # x0,y0,x1,y1 = picp
                # cx, cy = np.int(np.round((x0+x1)/2)), np.int(np.round((y0+y1)/2))
        #             print('0', cx, cy)
        #             draw_patch(pid, cx, cy)
        else:  # get iou
            picgs = np.array(pid_cat_gt[["x_min", "y_min", "x_max", "y_max"]])
            for picp in pid_cat_pred_xyxy:
                is_fp = True
                for picg in picgs:
                    temp_iou = calc_iou(picp, picg)
                    if temp_iou > 0:
                        is_fp = False
                if is_fp:
                    false_positives.append((pid, picp))
                # x0,y0,x1,y1 = picp
                # cx, cy = np.int(np.round((x0+x1)/2)), np.int(np.round((y0+y1)/2))
    #             print('iou', cx, cy)
    #             draw_patch(pid, cx, cy)

    return false_positives


def make_fp_patch(false_positives):
    num = 0
    pid0 = false_positives[0][0]
    for pid, (x0, y0, x1, y1) in tqdm(false_positives):

        if pid != pid0:
            num = 0
            pid0 = pid

        img = cv2.imread(f"{DATA_DIR}/png_1024l/train/{pid}.png", -1)
        cx = int(np.round((x0 + x1) / 2))
        cy = int(np.round((y0 + y1) / 2))
        py0 = max(0, cy - 127)
        px0 = max(0, cx - 127)
        py1 = min(1024, cy + 128)
        px1 = min(1024, cx + 128)

        patch = img[py0:py1, px0:px1]

        if patch.shape != (256, 256):
            patch = cv2.resize(patch, (256, 256))

        patch_name = f"{DATA_DIR}/patch_256/{finding[1]}/{pid}_fp_{num}.png"
        is_saved = cv2.imwrite(patch_name, patch)
        if not is_saved:
            raise ("not saved!")
        num += 1


#     break
# TODO:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/data2/minki/kaggle/vinbigdata-cxr"
    )
    parser.add_argument("--label", type=int, required=True)
    opt = parser.parse_args()

    FINDINGS = [
        "aortic_enlargement",  ### 0 614
        "atelectasis",  ### 1 37
        "calcification",  ### 2 90
        "cardiomegaly",  ### 3 460
        "consolidation",  ### 4 71
        "ild",  ### 5 77
        "infiltration",  ### 6 123
        "lung_opacity",  ### 7 264
        "nodule_mass",  ### 8 165
        "other_lesion",  ### 9 227
        "pleural_effusion",  ### 10 206
        "pleural_thickening",  ### 11 396
        "pneumothorax",  ### 12 20
        "pulmonary_fibrosis",  ### 13 323
        "no_finding",  ### 14 2121
    ]

    global DATA_DIR
    DATA_DIR = opt.data_dir
    train_df = pd.read_csv(f"{DATA_DIR}/yolov5/new_train.csv")
    # finding = (2, "calcification")
    finding = (opt.label, FINDINGS[opt.label])
    make_gt_patch(train_df, finding)

    false_positives = get_false_positives(train_df, finding)
    make_fp_patch(false_positives)
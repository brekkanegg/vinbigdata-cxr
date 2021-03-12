"""
Region을 고려에서 제외하는 경우(2048x2048 이미지 기반, pixel spacing=0.2mm)

"""

import numpy as np
from skimage.measure import regionprops, label
import cv2


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


def calc_iou_region(region_a, region_b):

    r_a = region_a._label_image == region_a.label
    r_b = region_b._label_image == region_b.label

    inter_ab = np.sum(r_a * r_b)

    return inter_ab / (region_a.area + region_b.area - inter_ab)


# Detection
def evaluate(cfgs, pred_bbox, gt_bbox):
    """
    :param pred: Prediction bbox Map, shape = (bbox_num, x0, x1, y0, y1, class, score)
    :param gt: Ground-truth Seg Map, shape = (bbox_num, x0, x1, y0, y1, class)
    :param iou_th: Threshold for prediction and gt matching
    :return:
        gt_nums: Ground-truth region numbers
        pred_nums: Prediction region numbers
        tp_nums: True Positive region numbers
        fp_nums: False Positive region numbers

    # 필수 가정: batch_size=1 (regionprops 함수가 2차원 행렬에만 적용 가능함)
    # Region을 고려에서 제외하는 경우(2048x2048 이미지 기반, pixel spacing=0.2mm)
    # i) Region bbox 크기 < 400 pixels
    # ii) (현재 사용x) Region bbox 장축<4mm(20pixels), 단축<2mm(10 pixels)

    # issue:  # 3. 영상사이즈는 디텍터 크기에 따라 달라질 수 있습니다. 완벽히 하기 위해선 pixel spacing 정보를 받아야 합니다.
    #         # 따라서 영상 크기에 대해 기준이 변경되는 것은 현단계에서는 적용할 필요가 없어 보입니다.
    """

    num_classes = cfgs["model"]["inputs"]["num_classes"]
    # image_size = cfgs["model"]["inputs"]["image_size"]

    if cfgs["run"] == "train":
        iou_th = 0.4
        prob_th = 0.5
    elif cfgs["run"] == "test":
        iou_th = 0.4
        prob_th = 0.5

    # 초기화
    # gt_nums = np.array([len(gt_bbox[gt_bbox[:, -1] == c]) for c in range(num_classes)])
    gt_nums = np.zeros(num_classes)
    tp_nums = np.zeros(num_classes)
    pred_nums = np.zeros(num_classes)
    fp_nums = np.zeros(num_classes)

    # Gt-Pred Bbox Iou Matrix
    for c in range(num_classes):
        c_gt_bbox = gt_bbox[gt_bbox[:, 4] == c]
        c_pred_bbox = pred_bbox[pred_bbox[:, 4] == c]

        thi_c_pred_bbox = c_pred_bbox[c_pred_bbox[:, -1] >= prob_th]

        gt_nums[c] = len(c_gt_bbox)
        pred_nums[c] = len(thi_c_pred_bbox)
        fp_nums[c] = len(thi_c_pred_bbox)

        iou_matrix = np.zeros((len(c_gt_bbox), len(thi_c_pred_bbox)))

        # FIXME: class
        for gi, gr in enumerate(c_gt_bbox):
            for pi, pr in enumerate(thi_c_pred_bbox):
                # BBox-IoU based
                iou_matrix[gi, pi] = calc_iou(gr, pr)
                # Region-IoU based
                # iou_matrix[gi, pi] = calc_iou_region(gr, pr)

        tp_nums[c] = np.sum(np.any((iou_matrix >= iou_th), axis=1))
        fp_nums[c] -= np.sum(np.any((iou_matrix > iou_th), axis=0))

    return gt_nums, pred_nums, tp_nums, fp_nums

import numpy as np


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


def wbf(bboxes_coord, bboxes_cat, iou_th=0.4):
    pass


def nms_savecat(bboxes_coord, bboxes_cat, iou_th=0.5):
    bbox_sizes = np.array([(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in bboxes_coord])
    order = bbox_sizes.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            if bboxes_cat[order[i]] != bboxes_cat[order[j]]:
                continue

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

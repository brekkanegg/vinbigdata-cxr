import argparse
import os, cv2
import numpy as np
import time
from copy import deepcopy
from tqdm import tqdm
from pprint import pprint

from ensemble_boxes import weighted_boxes_fusion
from glob import glob

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


def nms_svcat(bboxes_coord, bboxes_cat, bboxes_score, iou_thr=0.4):
    order = bboxes_score.argsort()[::-1]
    keep = [True] * len(order)

    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            if bboxes_cat[order[i]] != bboxes_cat[order[j]]:
                continue

            ov = calc_iou(bboxes_coord[order[i]], bboxes_coord[order[j]])
            if ov > iou_thr:
                keep[order[j]] = False

    bboxes_coord = [bb for (idx, bb) in enumerate(bboxes_coord) if keep[idx]]
    bboxes_cat = [bb for (idx, bb) in enumerate(bboxes_cat) if keep[idx]]
    bboxes_score = [bb for (idx, bb) in enumerate(bboxes_score) if keep[idx]]

    return bboxes_coord, bboxes_cat, bboxes_score

def lungseg_filtering(strings, uid, vote=False):
    outliers = [
        "7e5b17cc5c955fc4f8e9a626ab69295e",
        "285becd123b1dd66f5ce5e51bce073a4",
        "2127bbf73b1d0f150ca12bfa5ca2b91d",
        "7158cb1a21c2af6490567b0fe1dfaae0",
        "5118cbe9894006f58312565294e85556",
        "12202997d3fc0d506d2514a4c1995d15",
        "83918012739ab2906d692171f66a4dcc",
        "a3b5403784ffb7faa00003009aaa1d45",
        "b4e78fa47b3675c8de1ceae320bc4bf9",
        "cd8b2bb9706366aeaa560d62d888d4c0",
        "d32de0af32631052017592ebaaa11ce6",
        "d62af23bce04158a811760245247521f",
        "dd14fbccb38c064bdacc33391d8cca9e",
        "f599fe9d73fb3cc607a03fb258b7e198",
    ]
    try:
        if uid in outliers:
            return strings, [], None
        lungseg = cv2.imread(lungseg_dir + f"/{uid}_abdomen.png") // 255
        flags = [True] * len(strings)
        for idx, string in enumerate(strings):
            if string[0] in ["9", "14", "15"]:
                continue
            pcx = int(np.round(float(string[1]) * lungseg.shape[0]))
            pcy = int(np.round(float(string[2]) * lungseg.shape[1]))
            if vote:
                try:
                    voter = lungseg[pcy - 3 : pcy + 4, pcx - 3 : pcx + 4, 0]
                    if np.count_nonzero(voter) / np.prod(voter.shape) < 0.5:
                        if RULEOUT_VERBOSE:
                            print("Lungseg out", string)
                        flags[idx] = False
                        continue
                except:
                    if RULEOUT_VERBOSE:
                        print("Voting failed", string)
            if lungseg[pcy, pcx, 0] == 0:
                if RULEOUT_VERBOSE:
                    print("Lungseg out", string)
                flags[idx] = False
        return (
            [x for (_idx, x) in enumerate(strings) if flags[_idx]],
            [x for (_idx, x) in enumerate(strings) if not flags[_idx]],
            lungseg,
        )
    except Exception as e:
        print(e)
        return strings, [], None


def get_best_one(strings, targets=[0, 3, 14, 15]):
    if len(strings) < 2:
        return strings
    strs = deepcopy(strings)
    for target in targets:
        sub_strs = [x for x in strs if int(x[0]) == target]
        if len(sub_strs) > 1:
            best_prob = np.max([float(x[1]) for x in strs if int(x[0]) == target])
            strs = [
                x
                for x in strs
                if (int(x[0]) != target) or (np.abs(float(x[1]) - best_prob) < 1e-8)
            ]

        else:
            continue
    if (len(strs) != len(strings)) and RULEOUT_VERBOSE:
        print("Doubled")

    return strs


def rule_out_nearborder(strings, trbl=[0.03, 0.97, 0.97, 0.03]):
    if len(strings) < 1:
        return strings
    flags = [True] * len(strings)
    for idx, string in enumerate(strings):
        if string[0] == "9":
            continue

        if (
            float(string[1]) > trbl[1]
            or float(string[1]) < trbl[3]
            or float(string[2]) < trbl[0]
            or float(string[2]) > trbl[2]
        ):
            if RULEOUT_VERBOSE:
                print("NearBorder", string)
            flags[idx] = False

    return [x for (_idx, x) in enumerate(strings) if flags[_idx]]


def rule_out_elongated(strings, other_lesion=0.7, standard=0.8):
    if len(strings) < 1:
        return strings
    flags = [True] * len(strings)
    for idx, string in enumerate(strings):
        if string[0] in ["14", "15"]:
            continue
        threshold = standard if string[0] != "9" else other_lesion
        if (float(string[3]) > threshold) or (float(string[4]) > threshold):
            if RULEOUT_VERBOSE:
                print("Elongated", string)
            flags[idx] = False

    return [x for (_idx, x) in enumerate(strings) if flags[_idx]]


def rule_out_trivial(strings):
    # 계속 바뀔 예정
    if len(strings) < 1:
        return strings
    flags = [True] * len(strings)
    for idx, string in enumerate(strings):
        if string[0] == "0":  #
            if float(string[3]) > 2 * float(string[4]):
                if RULEOUT_VERBOSE:
                    print("Aortic w/h ratio", string)
                flags[idx] = False

        if string[0] == "2":  # calcification
            if float(string[4]) * float(string[3]) > 0.09:
                if RULEOUT_VERBOSE:
                    print("Calcification aratio", string)
                flags[idx] = False

        if string[0] == "3":  # Cardiomegaly
            if float(string[4]) > float(string[3]):
                if RULEOUT_VERBOSE:
                    print("Cardiomegaly w/h ratio", string)
                flags[idx] = False

    return [x for (_idx, x) in enumerate(strings) if flags[_idx]]


def apply_classwise_wbf(strings, cids=["10", "11", "13"], iou_thrs=[0.15, 0.15, 0.25], fn='nms'):
    if len(strings) < 1:
        return strings

    def _centroid2xyxy(bboxes):
        """
        yolo => [xmid, ymid, w, h] (normalized)
        voc  => [x1, y1, x2, y1]
        """
        bboxes = bboxes.copy().astype(float)
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]]
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]]
        bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
        bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
        return bboxes

    def _xyxy2centroid(bboxes):
        """
        voc  => [x1, y1, x2, y1]
        yolo => [xmid, ymid, w, h] (normalized)
        """
        bboxes = bboxes.copy().astype(float)
        bboxes[..., [2, 3]] = bboxes[..., [2, 3]] - bboxes[..., [0, 1]]
        bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]] / 2
        return bboxes

    dontcare = [x for x in strings if not x[0] in cids]
    for cid, iou_thr in zip(cids, iou_thrs):
        clswise_annot = [x for x in strings if x[0] == cid]
        if len(clswise_annot) < 2:
            dontcare += clswise_annot
            continue

        boxes = np.array([x[1:5] for x in clswise_annot], np.float32)
        lbls = np.array([cid] * len(clswise_annot))
        scores = np.array([x[5:] for x in clswise_annot], np.float32)

        boxes = _centroid2xyxy(boxes)

        # FIXME:
        if fn == 'nms':
            boxes, lbls, scores = nms_svcat(boxes, lbls, scores, iou_thr=iou_thr)
        elif fn == 'wbf':
            boxes, scores, lbls = weighted_boxes_fusion([boxes], [scores], [lbls], iou_thr=iou_thr)  # , conf_type="box_and_model_avg")

        boxes = _xyxy2centroid(boxes)
        aft_wbf = np.concatenate(
            [lbls[:, np.newaxis], boxes, scores[:, np.newaxis]], axis=-1
        )
        if (scores > 1).any():
            raise ("score larger than 1")

        for each_str in aft_wbf:
            dontcare += [
                "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    int(each_str[0]), *[x for x in each_str[1:]]
                ).split(" ")
            ]

    return dontcare


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--server", type=str, default="51")
    parser.add_argument("--wbf_all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    opt = parser.parse_args()
    global RULEOUT_VERBOSE
    RULEOUT_VERBOSE = 1 if opt.verbose else 0
    if opt.wbf_all:
        cids = [f"{i}" for i in range(14)]
        iou_thrs = [
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.15,
            0.15,
            0.4,
            0.25,
        ]

    else:
        cids = ["10", "11", "13"]
        iou_thrs = [0.15, 0.15, 0.25]

    pprint(opt)
    for cid, iou_thr in zip(cids, iou_thrs):
        print(cid, ":", iou_thr)

    global lungseg_dir
    if opt.server == "53":
        lungseg_dir = "/data/minki/kaggle/vinbigdata-cxr/final_segmap"
    elif opt.server == "51":
        lungseg_dir = "/data2/jryoungw/vinbigdata_cxr/final_segmap"

    print()
    time.sleep(1)
    target_labels = glob(f"{opt.src}/*.txt")
    path_to_revised = opt.dst
    os.makedirs(opt.dst, exist_ok=False)
    seg_out, rule_out, clswise_wbf = True, False, True
    nb_orig, nb_boxes, nb_segout, nb_ruledout, nb_wbfeffect = 0, 0, 0, 0, 0
    bef_cls_boxes = {str(cid): 0 for cid in range(16)}
    aft_cls_boxes = {str(cid): 0 for cid in range(16)}
    for idx_lbl, lbls in enumerate(target_labels):
        uid = lbls.split("/")[-1].split(".")[0]
        with open(lbls, "r") as f:
            strings = f.readlines()
            strings = [x.replace("\n", "").split(" ") for x in strings]

        bef_seg = len(strings)
        for strs in strings:
            bef_cls_boxes[strs[0]] += 1

        nb_orig += len([x for x in strings if not x[0] in ["14", "15"]])
        if seg_out:
            strings, segout, segmap = lungseg_filtering(strings, uid, False)
            nb_segout += len(segout)

        aft_seg = len(strings)
        if rule_out:
            strings = rule_out_trivial(strings)
            strings = get_best_one(strings)
            strings = rule_out_nearborder(strings)
            strings = rule_out_elongated(strings)

        aft_rule = len(strings)
        nb_ruledout += aft_seg - aft_rule
        if clswise_wbf:
            strings = apply_classwise_wbf(strings, cids=cids, iou_thrs=iou_thrs)

        aft_wbf = len(strings)
        nb_wbfeffect += aft_rule - aft_wbf
        nb_boxes += len([x for x in strings if not x[0] in ["14", "15"]])
        if os.path.exists(os.path.join(path_to_revised, uid + ".txt")):
            os.remove(os.path.join(path_to_revised, uid + ".txt"))

        with open(os.path.join(path_to_revised, uid + ".txt"), "a") as f:
            for strs in strings:
                aft_cls_boxes[strs[0]] += 1
                _strs = " ".join(strs) + "\n"
                f.write(_strs)

        print(
            "[{:04d}/{}] ".format(idx_lbl + 1, len(target_labels))
            + f"{uid} - original : {bef_seg}"
            + f", aft_segout : {aft_seg}"
            + f", aft_ruleout : {aft_rule}"
            + f", aft_wbf : {aft_wbf}"
        )

    print("-" * 100)
    print(bef_cls_boxes)
    print("total_segout    :", nb_segout)
    print("total_ruledout  :", nb_ruledout)
    print("total_wbfeffect :", nb_wbfeffect)
    print("=" * 100)

    print("final_bboxes    :", nb_boxes, " (without cid 14, 15)")
    print(aft_cls_boxes)
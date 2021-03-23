import os, sys
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pprint
import copy


import utils
from utils import misc
from inputs import vin
import models
import opts

from metrics.mapeval import VinBigDataEval


class Validator(object):
    def __init__(self, cfgs, device=None):

        self.cfgs = cfgs
        self.cfgs_val = cfgs["meta"]["val"]
        self.device = device

        ####### DATA
        self.val_loader = vin.get_dataloader(self.cfgs, mode="val")
        self.meta_dict = self.val_loader.dataset.meta_dict

        # Vin Eval
        self.nms_fn = self.val_loader.dataset.nms_fn

        # init
        self.gt_dict = None
        self.vineval = None

    def load_model(self):
        self.cfgs["save_dir"] = misc.set_save_dir(self.cfgs)
        self.tb_writer = utils.get_writer(self.cfgs)

        model = models.get_model(self.cfgs, pretrained=False)
        self.device = torch.device(f"cuda:{self.cfgs['local_rank']}")
        model = model.to(self.device)

        with open(os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb") as f:
            tot_val_record = pickle.load(f)
            resume_epoch = tot_val_record["best"]["epoch"]
            load_model_dir = os.path.join(
                self.cfgs["save_dir"], f"epoch_{resume_epoch}.pt"
            )
            checkpoint = torch.load(load_model_dir)
            model.load_state_dict(checkpoint["model"], strict=True)
            # self.txt_logger.write("\n\nValidate Here! \n\n")

        return model

    # get dictionary with nms bbox results
    def get_gt_dict(self):

        ims = self.cfgs["meta"]["inputs"]["image_size"]
        nms_iou = self.cfgs["meta"]["model"]["nms_iou"]

        temp_dict = {}
        for pid in tqdm(self.val_loader.dataset.pids):

            bboxes_coord = []
            bboxes_cat = []
            bboxes_rad = []

            pid_info = self.meta_dict[pid]
            pid_dimy = pid_info["dim0"]
            pid_dimx = pid_info["dim1"]

            pid_bbox = np.array(pid_info["bbox"])
            pid_rad = pid_bbox[:, 0]

            if self.cfgs["meta"]["inputs"]["cat"] is not None:
                cid = self.cfgs["meta"]["inputs"]["cat"]
                cat_idx = [True if i == cid else False for i in pid_bbox[:, 2]]
                pid_bbox = pid_bbox[cat_idx]
                pid_label = pid_label[cat_idx]
                pid_rad = pid_rad[cat_idx]

            # pid_bbox order: rad_id, finding, finding_id, bbox(x_min, y_min, x_max, y_max) - xyxy가로, 세로
            for bi, bb in enumerate(pid_bbox):
                bx0, by0, bx1, by1 = [float(i) for i in bb[-4:]]
                blabel = int(bb[2])
                brad = int(pid_rad[bi])
                if blabel == 14:
                    continue

                if (bx0 >= bx1) or (by0 >= by1):
                    continue
                else:
                    temp_bb = [None, None, None, None]
                    temp_bb[0] = np.round(bx0 / pid_dimx * ims)
                    temp_bb[1] = np.round(by0 / pid_dimy * ims)
                    temp_bb[2] = np.round(bx1 / pid_dimx * ims)
                    temp_bb[3] = np.round(by1 / pid_dimy * ims)

                    bboxes_coord.append(temp_bb)
                    bboxes_cat.append(blabel)
                    bboxes_rad.append(brad)

            if len(bboxes_coord) >= 2:
                bboxes_coord, bboxes_cat = self.nms_fn(
                    bboxes_coord, bboxes_cat, bboxes_rad, nms_iou, ims
                )

            bboxes = [list(b) + [c] for (b, c) in zip(bboxes_coord, bboxes_cat)]
            bboxes = np.array(bboxes).astype(int)

            if len(bboxes) == 0:
                temp_dict[pid] = {"bbox": np.array([[0, 0, 1, 1, 14]])}
            else:
                temp_dict[pid] = {"bbox": bboxes}

        return temp_dict

    def do_validate(self, model=None):

        if self.gt_dict is None:
            self.gt_dict = self.get_gt_dict()
            self.vineval = VinBigDataEval(self.gt_dict)

        self.pred_dict = {}

        if model is not None:
            self.model = model.to(self.device)
        else:
            if self.cfgs["run"] == "val":
                self.model = self.load_model()

        ####### Init val result
        nums_tot = 0
        losses_tot = 0
        dlosses_tot = 0
        closses_tot = 0

        # Classification metrics
        cls_pred_tot = []
        cls_gt_tot = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():

            tqdm_able = self.cfgs["run"] != "train"
            for data in tqdm(self.val_loader, disable=(not tqdm_able)):

                img = data["img"].permute(0, 3, 1, 2).to(self.device)

                logits = self.model(img, mode="val")

                dloss, closs = opts.calc_loss(self.cfgs, self.device, data, logits)
                loss = dloss + self.cfgs["meta"]["loss"]["cls_weight"] * closs

                losses_tot += loss * len(data["fp"])
                dlosses_tot += dloss * len(data["fp"])
                closses_tot += closs * len(data["fp"])
                nums_tot += len(data["fp"])

                det_anns = data["bbox"].numpy()

                viz_bi = 0
                for bi in range(len(data["fp"])):
                    bi_fp = data["fp"][bi]

                    # Prediciton
                    bi_det_preds = logits["preds"][bi]  # .detach().cpu().numpy()
                    bi_det_preds = bi_det_preds[bi_det_preds[:, -1] != -1]

                    # GT
                    bi_det_anns = det_anns[bi]
                    bi_det_anns = bi_det_anns[bi_det_anns[:, -1] != -1]

                    if len(bi_det_preds) == 0:  # No pred bbox
                        bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])
                    else:
                        bi_det_preds = bi_det_preds.detach().cpu().numpy()
                        bi_det_preds[:, :4] = np.round(bi_det_preds[:, :4]).astype(int)

                    # if not self.cfgs["meta"]["inputs"]["abnormal_only"]:
                    bi_cls_pred = torch.sigmoid(logits["aux_cls"][bi]).item()
                    # if self.cfgs_val["use_classifier"]:
                    if bi_cls_pred < self.cfgs_val["cls_th"]:
                        bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    cls_pred_tot.append(bi_cls_pred)
                    cls_gt_tot.append(int(len(bi_det_anns) > 0))

                    self.pred_dict[bi_fp] = {"bbox": bi_det_preds}

                    if len(bi_det_anns) == 0:  # No pred bbox
                        # This is dummy bbox
                        bi_det_anns = np.array([[0, 0, 1, 1, 14]])

                    # Save_png
                    if (self.cfgs["run"] == "val") and self.cfgs_val["save_png"]:
                        self.tb_writer.write_images(
                            bi_fp,
                            data["img"][bi].numpy(),
                            bi_det_preds,
                            bi_det_anns,
                            0,
                            "val",
                            save=True,
                        )

            # For Visualization in TB - abnormal
            # FIXME:
            if self.cfgs["run"] == "train":
                vizlist = np.random.permutation(list(range(len(data["fp"]))))
                for viz_bi in vizlist:
                    if data["bbox"][viz_bi, 0, -1] != -1:
                        break

                det_preds_viz = logits["preds"][viz_bi].detach().cpu().numpy()
                if len(det_preds_viz) == 0:  # No pred bbox
                    det_preds_viz = np.ones((1, 6)) * -1

                det_anns_viz = data["bbox"][viz_bi].detach().cpu().numpy()

                val_viz = {
                    "fp": data["fp"][viz_bi],
                    "img": data["img"][viz_bi].numpy(),
                    "pred": det_preds_viz,
                    "ann": det_anns_viz,
                }

        cls_gt_tot = np.array(cls_gt_tot)
        cls_pred_tot = np.array(cls_pred_tot)
        tn, fp, fn, tp = confusion_matrix(
            cls_gt_tot, cls_pred_tot > self.cfgs["meta"]["val"]["cls_th"]
        ).ravel()
        cls_sens = tp / (tp + fn + 1e-5)
        cls_spec = tn / (tn + fp + 1e-5)
        cls_auc = roc_auc_score(cls_gt_tot, cls_pred_tot)

        val_record = {
            "loss": (losses_tot / (nums_tot + 1e-5)),
            "dloss": (dlosses_tot / (nums_tot + 1e-5)),
            "closs": (closses_tot / (nums_tot + 1e-5)),
            "cls_auc": cls_auc,
            "cls_sens": cls_sens,
            "cls_spec": cls_spec,
        }

        if self.cfgs["meta"]["train"]["samples_per_epoch"] is not None:
            self.vineval.image_ids = sorted(self.pred_dict.keys())

        # TODO:
        if self.cfgs["meta"]["val"]["clsth_search"]:
            cls_th_combi = self.cls_th_search()

            dths = [i[0] for i in cls_th_combi]
            APs = [i[1] for i in cls_th_combi]
            mAP = np.mean(APs)
            val_record["mAP"] = mAP
            val_record["APs"] = APs
            val_record["dths"] = dths

        else:
            self.vineval.predictions["annotations"] = self.vineval.gen_predictions(
                self.pred_dict
            )

            coco_eval = self.vineval.evaluate()
            mAP, APs = coco_eval.stats
            val_record["mAP"] = mAP
            val_record["APs"] = APs

        # if not self.cfgs["meta"]["inputs"]["abnormal_only"]:

        if self.cfgs["run"] == "val":
            pprint.pprint(val_record)

        elif self.cfgs["run"] == "train":
            return val_record, val_viz

    def cls_th_search(self):
        # dths = [0.01, 0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.48]
        dths = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

        cls_th_combi = []
        for c in tqdm(range(15)):
            c_vineval = copy.deepcopy(self.vineval)
            c_vineval.annotations["categories"] = [
                c_vineval.annotations["categories"][c]
            ]
            c_vineval.annotations["annotations"] = [
                r
                for r in self.vineval.annotations["annotations"]
                if r["category_id"] == c
            ]
            c_vineval.predictions["categories"] = [
                c_vineval.predictions["categories"][c]
            ]

            # Threshold-wise:
            temp_dict = copy.deepcopy(self.pred_dict)
            # FIXME: if len 0 remove?
            for k in self.pred_dict.keys():
                temp_dict[k]["bbox"] = temp_dict[k]["bbox"][
                    temp_dict[k]["bbox"][:, 4] == c
                ]

            dths_aps = []
            for dth in dths:
                temp_dict2 = copy.deepcopy(temp_dict)
                for k in self.pred_dict.keys():
                    temp_dict2[k]["bbox"] = temp_dict2[k]["bbox"][
                        temp_dict2[k]["bbox"][:, 5] >= dth
                    ]
                c_vineval.predictions["annotations"] = c_vineval.gen_predictions(
                    temp_dict2
                )
                c_eval = c_vineval.evaluate()
                c_AP, _ = c_eval.stats
                dths_aps.append(c_AP)

            # print(dths_aps)
            max_idx = np.argmax(dths_aps)
            max_AP = dths_aps[max_idx]
            max_dth = dths[max_idx]
            cls_th_combi.append([max_dth, max_AP])

        return cls_th_combi


# 0. image classifier 의 값이 0.003751 이하면 prediction 없음
# 0-1. 위가 0.003751 이하이어도 bbox score가 0.95 이상이면 사용
# 1. 0:Aortic Enlargment, 3:Cardiomegaly 는 이미지당 한 개만
# 2. 11:Pleural_thickening 의 경우는 0.015 이상만
# 3. 9:Other_lesion 의 경우는 0.1 이상
# 4. 레이블 고려해서 nms 사용(iou_th=0.4)
# 4-1. 2:calcification, 11-pleural thicknening 은 iou_th=0.001

# for box_id in range(len(detect_result) // 6)[::-1]:
#     label, *box, score = detect_result[6 * box_id : 6 * box_id + 6]
#     if class_labels.item() >= self.config.classification_thresh:
#         if (
#             (score > self.config.score_last)  # 0
#             and not (label in [0, 3] and label in list_label)
#             and not (
#                 label == 11 and score < self.config.score_11
#             )  # pleural thickening, 0.015
#             and not (label == 9 and score < self.config.score_9)  # other-lesion 0.1
#         ):
#             list_label.append(label)
#             box = label_resize(self.config.img_size, img_size, *box)
#             result_one_image.append(int(label))
#             result_one_image.append(np.round(score, 3))
#             result_one_image.extend([int(i) for i in box])

#         else:
#             if (
#                 score > self.config.score_last2  # 0.95
#                 and not (label in [0, 3] and label in list_label)
#                 and not (label == 11 and score < self.config.score_11)
#                 and not (label == 9 and score < self.config.score_9)
#             ):
#                 list_label.append(label)
#                 box = label_resize(self.config.img_size, img_size, *box)
#                 result_one_image.append(int(label))
#                 result_one_image.append(np.round(score, 3))
#                 result_one_image.extend([int(i) for i in box])


# def label_process(self, detect_result, iou_thresh, iou_thresh11):
#     assert detect_result != ""
#     x_center, y_center = detect_result[1::6], detect_result[2::6]
#     w_center, h_center = detect_result[3::6], detect_result[4::6]
#     detect_result[1::6] = [i - 0.5 * j for i, j in zip(x_center, w_center)]
#     detect_result[2::6] = [i - 0.5 * j for i, j in zip(y_center, h_center)]
#     detect_result[3::6] = [i + 0.5 * j for i, j in zip(x_center, w_center)]
#     detect_result[4::6] = [i + 0.5 * j for i, j in zip(y_center, h_center)]
#     list_new = []

#     for label_values in np.unique(detect_result[::6]):
#         list_values = np.array(
#             [
#                 detect_result[6 * idx : 6 * idx + 6]
#                 for idx, i in enumerate(detect_result[::6])
#                 if i == label_values
#             ]
#         )
#         boxes = list_values[:, 1:5].tolist()
#         scores = list_values[:, 5].tolist()
#         labels = list_values[:, 0].tolist()
#         if label_values in [2, 11]:
#             boxes, scores, labels = nms(
#                 [boxes], [scores], [labels], weights=None, iou_thr=iou_thresh11  # 0.001
#             )
#         else:
#             boxes, scores, labels = nms(
#                 [boxes], [scores], [labels], weights=None, iou_thr=iou_thresh  # 0.4
#             )

#         for box in list_values:
#             if box[-1] in scores:
#                 list_new.extend(box)
#     return list_new

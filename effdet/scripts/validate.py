import os, sys
import torch
import numpy as np
import pandas as pd
import pickle
import cv2
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import utils
import inputs
import opts
from metrics.metrics import evaluate
from metrics.cocoeval import VinBigDataEval


class Validator(object):
    def __init__(self, cfgs, device):

        self.cfgs = cfgs
        self.device = device

        ####### DATA
        val_transforms = None  # FIXME: TTA
        val_dataset = inputs.get_dataset(self.cfgs, mode="val")
        val_sampler = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfgs["batch_size"] * 4,
            num_workers=cfgs["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=inputs.get_collater(),
            sampler=val_sampler,
        )

        self.meta_dict = self.val_loader.dataset.meta_dict
        self.ims = self.cfgs["model"]["inputs"]["image_size"]

        self.do_logging = True
        if len(self.cfgs["gpu"]) > 1:
            if dist.get_rank() != 0:
                self.do_logging = False

        # Vin Eval
        self.gt_dict = self.get_gt_dict()
        self.vineval = VinBigDataEval(self.gt_dict)

    # get dictionary with nms bbox results
    def get_gt_dict(self):
        from inputs.vin import simple_nms

        temp_dict = {}
        for pid in tqdm(self.val_loader.dataset.pids):

            bboxes_coord = []
            bboxes_cat = []

            pid_info = self.meta_dict[pid]
            pid_bbox = np.array(pid_info["bbox"])
            # pid_bbox order: rad_id, finding, finding_id, bbox(x_min, y_min, x_max, y_max) - xyxy가로, 세로

            for bb in pid_bbox:
                bx0, by0, bx1, by1 = [float(i) for i in bb[-4:]]
                bl = int(bb[2])
                if bl == 14:
                    continue

                if (bx0 >= bx1) or (by0 >= by1):
                    continue
                else:
                    bboxes_coord.append([bx0, by0, bx1, by1])
                    bboxes_cat.append(bl)

            # FIXME: simple_nms
            if len(bboxes_coord) >= 2:
                bboxes_coord, bboxes_cat = simple_nms(bboxes_coord, bboxes_cat)

            bboxes = [list(b) + [c] for (b, c) in zip(bboxes_coord, bboxes_cat)]
            bboxes = np.array(bboxes).astype(int)

            if len(bboxes) == 0:
                temp_dict[pid] = {"bbox": np.array([[0, 0, 1, 1, 14]])}
            else:
                temp_dict[pid] = {"bbox": bboxes}

        return temp_dict

    def do_validate(self, model=None):

        self.pred_dict = {}

        self.model = model.to(self.device)

        ####### Init val result
        det_gt_nums_tot = np.zeros((15))
        det_tp_nums_tot = np.zeros((1, 15))
        det_fp_nums_tot = np.zeros((1, 15))
        det_pred_nums_tot = np.zeros((1, 15))
        nums_tot = 0
        losses_tot = 0
        dlosses_tot = 0
        closses_tot = 0

        # Classification metrics
        cls_pred_tot = []
        cls_gt_tot = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():

            tqdm_able = (self.cfgs["run"] != "train") and self.do_logging
            for data in tqdm(self.val_loader, disable=(not tqdm_able)):

                img = data["img"].permute(0, 3, 1, 2).to(self.device)

                logits = self.model(img, mode="val")
                dloss, closs = opts.calc_loss(self.cfgs, self.device, data, logits)
                loss = dloss + self.cfgs["model"]["loss"]["cls_weight"] * closs

                if len(self.cfgs["gpu"]) > 1:
                    loss = torch.mean(gather_helper(loss))
                    data["fp"] = gather_helper(data["fp"])
                    data["bbox"] = gather_helper(data["bbox"])
                    # FIXME:
                    logits["preds"] = gather_helper(logits["preds"])
                    logits["aux_cls"] = gather_helper(logits["aux_cls"])

                loss = loss.detach().item()
                dloss = dloss.detach().item()
                closs = closs.detach().item()
                losses_tot += loss * len(data["fp"])
                dlosses_tot += dloss * len(data["fp"])
                closses_tot += closs * len(data["fp"])
                nums_tot += len(data["fp"])

                det_anns = data["bbox"].numpy()

                viz_bi = 0
                for bi in range(len(data["fp"])):
                    bi_fp = data["fp"][bi]

                    # Prediciton
                    bi_det_preds = logits["preds"][bi].detach().cpu().numpy()
                    bi_det_preds = bi_det_preds[bi_det_preds[:, -1] != -1]

                    # TODO: use aux_classifier for cls_pred
                    if len(bi_det_preds) == 0:  # No pred bbox
                        # bi_det_preds = np.ones((1, 6)) * -1
                        bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    else:
                        bi_dim0 = self.meta_dict[bi_fp]["dim0"]
                        bi_dim1 = self.meta_dict[bi_fp]["dim1"]
                        bi_det_preds[:, [0, 2]] *= bi_dim0 / self.ims
                        bi_det_preds[:, [1, 3]] *= bi_dim1 / self.ims
                        bi_det_preds = np.round(bi_det_preds).astype(int)

                    bi_cls_pred = torch.sigmoid(logits["aux_cls"][bi][0]).item()
                    if self.cfgs["model"]["val"]["use_classifier"]:
                        if bi_cls_pred < self.cfgs["model"]["val"]["cls_th"]:
                            bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    self.pred_dict[bi_fp] = {"bbox": bi_det_preds}

                    # GT
                    bi_det_ann = det_anns[bi]
                    bi_det_ann = bi_det_ann[bi_det_ann[:, -1] != -1]
                    (
                        bi_det_gt_num,
                        bi_det_pred_num,
                        bi_det_tp_num,
                        bi_det_fp_num,
                    ) = evaluate(self.cfgs, bi_det_preds, bi_det_ann)

                    det_gt_nums_tot += bi_det_gt_num
                    det_tp_nums_tot += bi_det_tp_num
                    # correct_nums_tot += correct_num
                    det_pred_nums_tot += bi_det_pred_num
                    det_fp_nums_tot += bi_det_fp_num

                    cls_pred_tot.append(bi_cls_pred)
                    cls_gt_tot.append(int(len(bi_det_ann) > 0))

            # for Visualization - abnormal
            for viz_bi in range(len(data["fp"])):
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

        coco_evaluation = self.vineval.evaluate(self.pred_dict)

        det_pc = det_tp_nums_tot / (det_pred_nums_tot + 1e-5)
        det_rc = det_tp_nums_tot / (det_gt_nums_tot + 1e-5)
        det_fppi = det_fp_nums_tot / (nums_tot + 1e-5)

        cls_gt_tot = np.array(cls_gt_tot)
        cls_pred_tot = np.array(cls_pred_tot)

        try:
            tn, fp, fn, tp = confusion_matrix(cls_gt_tot, cls_pred_tot > 0.5).ravel()
            cls_sens = tp / (tp + fn + 1e-5)
            cls_spec = tn / (tn + fp + 1e-5)
            cls_auc = roc_auc_score(cls_gt_tot, cls_pred_tot)

        except Exception as e:
            cls_auc = 0
            cls_sens = 0
            cls_spec = 0

        # except class 14 - normal
        # det_pc = det_pc[0, :-1].mean()
        # det_rc = det_rc[0, :-1].mean()
        # det_fppi = det_fppi[0, :-1].mean()

        val_record = {
            "loss": (losses_tot / (nums_tot + 1e-5)),
            "dloss": (dlosses_tot / (nums_tot + 1e-5)),
            "closs": (closses_tot / (nums_tot + 1e-5)),
            "det_prec": det_pc,
            "det_recl": det_rc,
            "det_fppi": det_fppi,
            "det_f1": 2 * det_pc * det_rc / (det_pc + det_rc + 1e-5),
            "cls_auc": cls_auc,
            "cls_sens": cls_sens,
            "cls_spec": cls_spec,
            "coco": coco_evaluation.stats[0],
        }

        return val_record, val_viz


# def sigmoid_func(x):
#     return 1 / (1 + np.exp(-x))


def gather_helper(target):
    target_gather = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
    dist.all_gather(target_gather, target, async_op=False)
    target = torch.cat(target_gather, 0)

    return target


def pred2string(pred):
    string = ""
    for i in pred:
        string += f" {i}"
    return string
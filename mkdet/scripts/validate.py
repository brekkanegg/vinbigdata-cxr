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
from utils import misc
import inputs
from inputs import nms
import opts
from metrics.metrics import evaluate
from metrics.cocoeval import VinBigDataEval


class Validator(object):
    def __init__(self, cfgs, device=None):

        self.cfgs = cfgs
        self.cfgs_val = cfgs["model"]["val"]
        self.device = device

        ####### DATA
        val_transforms = None  # FIXME:
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

        # Vin Eval
        self.nms = self.val_loader.dataset.nms
        self.gt_dict = self.get_gt_dict()
        self.vineval = VinBigDataEval(self.gt_dict)

    def load_model(self):
        self.cfgs["save_dir"] = misc.set_save_dir(self.cfgs)
        self.tb_writer = utils.get_writer(self.cfgs)

        import models

        model = models.get_model(self.cfgs)
        self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
        model = model.to(self.device)

        with open(os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb") as f:
            tot_val_record = pickle.load(f)
            resume_epoch = tot_val_record["best"]["epoch"]
            load_model_dir = os.path.join(
                self.cfgs["save_dir"], "epoch_{}.pt".format(resume_epoch)
            )
            checkpoint = torch.load(load_model_dir)
            model.load_state_dict(checkpoint["model"], strict=True)
            # self.txt_logger.write("\n\nValidate Here! \n\n")

        return model

    # get dictionary with nms bbox results
    def get_gt_dict(self):

        ims = self.cfgs["model"]["inputs"]["image_size"]

        temp_dict = {}
        for pid in tqdm(self.val_loader.dataset.pids):

            bboxes_coord = []
            bboxes_cat = []

            pid_info = self.meta_dict[pid]
            pid_dimy = pid_info["dim0"]
            pid_dimx = pid_info["dim1"]

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
                    temp_bb = [None, None, None, None]
                    temp_bb[0] = np.round(bx0 / pid_dimx * ims)
                    temp_bb[1] = np.round(by0 / pid_dimy * ims)
                    temp_bb[2] = np.round(bx1 / pid_dimx * ims)
                    temp_bb[3] = np.round(by1 / pid_dimy * ims)

                    bboxes_coord.append(temp_bb)
                    bboxes_cat.append(bl)

            # FIXME: simple_nms
            if len(bboxes_coord) >= 2:
                bboxes_coord, bboxes_cat = self.nms(
                    bboxes_coord, bboxes_cat, iou_th=0.5
                )

            bboxes = [list(b) + [c] for (b, c) in zip(bboxes_coord, bboxes_cat)]
            bboxes = np.array(bboxes).astype(int)

            if len(bboxes) == 0:
                temp_dict[pid] = {"bbox": np.array([[0, 0, 1, 1, 14]])}
            else:
                temp_dict[pid] = {"bbox": bboxes}

        return temp_dict

    def do_validate(self, model=None):

        self.pred_dict = {}

        if model is not None:
            self.model = model.to(self.device)
        else:
            if self.cfgs["run"] == "val":
                self.model = self.load_model()

        ####### Init val result
        num_classes = self.cfgs["model"]["inputs"]["num_classes"]

        det_gt_nums_tot = np.zeros(num_classes)
        det_tp_nums_tot = np.zeros(num_classes)
        det_fp_nums_tot = np.zeros(num_classes)
        det_pred_nums_tot = np.zeros(num_classes)
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
                loss = dloss + self.cfgs["model"]["loss"]["cls_weight"] * closs

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
                        bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])
                    else:
                        bi_det_preds[:, :4] = np.round(bi_det_preds[:, :4]).astype(int)

                    bi_cls_pred = torch.sigmoid(logits["aux_cls"][bi]).item()
                    if self.cfgs_val["use_classifier"]:
                        if bi_cls_pred < self.cfgs_val["cls_th"]:
                            bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    cls_pred_tot.append(bi_cls_pred)

                    self.pred_dict[bi_fp] = {"bbox": bi_det_preds}

                    # GT
                    bi_det_anns = det_anns[bi]
                    bi_det_anns = bi_det_anns[bi_det_anns[:, -1] != -1]

                    cls_gt_tot.append(int(len(bi_det_anns) > 0))

                    if len(bi_det_anns) == 0:  # No pred bbox
                        # This is dummy bbox
                        bi_det_anns = np.array([[0, 0, 1, 1, 14]])

                    # evaluation
                    (
                        bi_det_gt_num,
                        bi_det_pred_num,
                        bi_det_tp_num,
                        bi_det_fp_num,
                    ) = evaluate(self.cfgs, bi_det_preds, bi_det_anns)

                    det_gt_nums_tot += bi_det_gt_num
                    det_tp_nums_tot += bi_det_tp_num
                    det_pred_nums_tot += bi_det_pred_num
                    det_fp_nums_tot += bi_det_fp_num

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

        if self.cfgs["run"] == "val":
            print(val_record)

        return val_record, val_viz

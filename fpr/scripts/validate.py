import os, sys
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pprint
import timm
import utils
from utils import misc
from inputs import patch


class Validator(object):
    def __init__(self, cfgs, device=None):

        self.cfgs = cfgs
        self.cfgs_val = cfgs["meta"]["val"]
        self.device = device

        ####### DATA
        self.val_loader = patch.get_dataloader(self.cfgs, mode="val")

        self.criterion = nn.BCEWithLogitsLoss()

    def load_model(self):
        self.cfgs["save_dir"] = misc.set_save_dir(self.cfgs)
        self.tb_writer = utils.get_writer(self.cfgs)

        mname = self.cfgs["meta"]["model"]["name"]

        model = timm.create_model(
            f"tf_efficientnet_{mname}_ns", pretrained=True, num_classes=1
        )

        self.device = torch.device(f"cuda:{self.cfgs['local_rank']}")
        model = model.to(self.device)

        with open(os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb") as f:
            tot_val_record = pickle.load(f)
            print(tot_val_record["best"])
            resume_epoch = tot_val_record["best"]["epoch"]
            # resume_epoch = 53
            load_model_dir = os.path.join(
                self.cfgs["save_dir"], f"epoch_{resume_epoch}.pt"
            )
            checkpoint = torch.load(load_model_dir)
            model.load_state_dict(checkpoint["model"], strict=True)

        return model

    def do_validate(self, model=None):

        if model is not None:
            self.model = model.to(self.device)
        else:
            if self.cfgs["run"] == "val":
                self.model = self.load_model()

        ####### Init val result
        nums_tot = 0
        losses_tot = 0

        # Classification metrics
        cls_pred_tot = []
        cls_gt_tot = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():
            for data in self.val_loader:
                # print(data["fp"])
                img = data["img"].permute(0, 3, 1, 2).to(self.device)

                logits = self.model(img)
                loss = self.criterion(logits, data["label"].to(self.device))

                losses_tot += loss * len(data["fp"])
                nums_tot += len(data["fp"])

                for bi in range(len(data["fp"])):
                    cls_pred_tot.append(torch.sigmoid(logits[bi]).item())
                    cls_gt_tot.append(float(data["label"][bi].item() > 0.5))

        cls_gt_tot = np.array(cls_gt_tot)
        cls_pred_tot = np.array(cls_pred_tot)
        tn, fp, fn, tp = confusion_matrix(
            cls_gt_tot, cls_pred_tot > self.cfgs["meta"]["val"]["cls_th"]
        ).ravel()
        cls_sens = tp / (tp + fn + 1e-5)
        cls_spec = tn / (tn + fp + 1e-5)
        cls_auc = roc_auc_score(cls_gt_tot, cls_pred_tot)

        fit_weight = np.array([0.5, 0.1, 0.4])
        fit = np.sum(np.array([cls_sens, cls_spec, cls_auc]) * fit_weight)

        val_record = {
            "loss": (losses_tot / (nums_tot + 1e-5)),
            "cls_auc": cls_auc,
            "cls_sens": cls_sens,
            "cls_spec": cls_spec,
            "fit": fit,
        }

        if self.cfgs["run"] == "val":
            pprint.pprint(val_record)

        elif self.cfgs["run"] == "train":
            return val_record

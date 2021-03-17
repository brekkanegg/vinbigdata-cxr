import os, sys
import torch
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
from utils import misc
import inputs
import pprint


class Testor(object):
    def __init__(self, cfgs, device=None):

        self.cfgs = cfgs
        self.cfgs_test = self.cfgs["meta"]["test"]
        self.device = device

        ####### DATA
        test_transforms = None  # FIXME:
        test_dataset = inputs.get_dataset(self.cfgs, mode="test")

        test_sampler = None
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfgs["batch_size"],
            num_workers=cfgs["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=inputs.get_collater("test"),
            sampler=test_sampler,
        )

        self.meta_dict = self.test_loader.dataset.meta_dict
        self.ims = self.cfgs["meta"]["inputs"]["image_size"]

    def load_model(self, load_dir, load_epoch=None):
        # self.cfgs["save_dir"] = misc.set_save_dir(self.cfgs)

        # import models
        if self.cfgs["meta"]["model"]["old"]:
            from models.efficientdet.model_outdated import EfficientDet
        else:
            from models.efficientdet.model import EfficientDet

        model = EfficientDet(self.cfgs)
        self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
        model = model.to(self.device)

        ckpt_dir = f"/nfs3/minki/kaggle/vinbigdata-cxr/mkdet/ckpt/{load_dir}"

        with open(ckpt_dir + "/tot_val_record.pkl", "rb") as f:
            tot_val_record = pickle.load(f)

        if load_epoch == "":
            best_epoch = tot_val_record["best"]["epoch"]

        else:
            best_epoch = load_epoch

        load_model_dir = os.path.join(ckpt_dir, "epoch_{}.pt".format(best_epoch))

        print("Load: ", load_model_dir)
        pprint.pprint(tot_val_record[str(best_epoch)])

        checkpoint = torch.load(load_model_dir)
        model.load_state_dict(checkpoint["model"], strict=True)

        return model

    def do_test(self):

        load_dir = input("Test ckpt dir: ")
        try:
            memo = input("Submit csv name: ")
        except SyntaxError:
            print("Enter submit csv name")
            raise ()
        load_epoch = input("Test epoch: ")
        reduce_size = input("Reduce bbox size (t/f): ")
        if reduce_size == "t":
            reduce_size = True
        else:
            reduce_size = False

        # use_classifier = input("Use classifier (t/f): ")
        # if use_classifier == "f":
        #     use_classifier = False
        # else:
        #     use_classifier = True

        print("\n\nCheck following: ")
        print("Image Size: ", self.cfgs["meta"]["inputs"]["image_size"])
        print(
            "Detection only - use abnormal only: ",
            self.cfgs["meta"]["inputs"]["abnormal_only"],
        )
        print(
            "Num Classes - include No Finding: ",
            self.cfgs["meta"]["inputs"]["num_classes"],
        )

        print("Doing Inference.. ")
        self.model = self.load_model(load_dir, load_epoch)

        submit_df = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():

            for data in tqdm(self.test_loader):
                img = data["img"].permute(0, 3, 1, 2).to(self.device)
                logits = self.model(img, mode="test")

                for bi in range(len(data["fp"])):
                    bi_fp = data["fp"][bi]

                    # Prediciton
                    bi_det_preds = logits["preds"][bi].detach().cpu().numpy()
                    bi_det_preds = bi_det_preds[
                        bi_det_preds[:, -1] >= self.cfgs_test["prob_th"]
                    ]

                    if len(bi_det_preds) == 0:  # No pred bbox
                        bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    else:
                        bi_dimy = self.meta_dict[bi_fp]["dim0"]
                        bi_dimx = self.meta_dict[bi_fp]["dim1"]
                        bi_det_preds[:, [0, 2]] *= bi_dimx / self.ims
                        bi_det_preds[:, [1, 3]] *= bi_dimy / self.ims
                        bi_det_preds[:, :4] = np.round(bi_det_preds[:, :4])

                    # FIXME:
                    if not self.cfgs["meta"]["inputs"]["abnormal_only"]:
                        bi_cls_pred = torch.sigmoid(logits["aux_cls"][bi]).item()

                        if bi_cls_pred < self.cfgs["meta"]["test"]["cls_th"]:
                            bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    pred_string = ""
                    for det_i in bi_det_preds:
                        x0, y0, x1, y1, c, cf = det_i
                        if reduce_size:
                            (x0, y0, x1, y1) = reduce_bbox((x0, y0, x1, y1))

                        pred_string += (
                            f" {int(c)} {cf} {int(x0)} {int(y0)} {int(x1)} {int(y1)}"
                        )

                    pred_string = pred_string[1:]  # remove left blank

                    submit_df.append([bi_fp, pred_string])

        # Make submit csv
        submit_csv = pd.DataFrame(submit_df, columns=["image_id", "PredictionString"])

        # Check number of normal row
        print("\n\nTotal Number of Rows: ", len(submit_csv))
        print(
            "Number of Normal Rows: ",
            len(submit_csv[submit_csv["PredictionString"] == "14 1 0 0 1 1"]),
        )

        submit_dir = os.path.join(self.cfgs_test["submit_dir"], f"{memo}_submit.csv")
        submit_csv.to_csv(submit_dir, index=False)
        print("Submission csv saved in: ", submit_dir)


def reduce_bbox(bbox, r=0.9):
    x0, y0, x1, y1 = bbox
    if bbox == (0, 0, 1, 1):
        return bbox

    rx = (x1 - x0) * r / 2
    ry = (y1 - y0) * r / 2
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    resized_bbox = (
        np.round(cx - rx),
        np.round(cy - ry),
        np.round(cx + rx),
        np.round(cy + ry),
    )

    return resized_bbox

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


class Testor(object):
    def __init__(self, cfgs, device=None):

        self.cfgs = cfgs
        self.cfgs_test = self.cfgs["model"]["test"]
        self.device = device

        ####### DATA
        test_transforms = None  # FIXME:
        test_dataset = inputs.get_dataset(self.cfgs, mode="test")

        test_sampler = None
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfgs["batch_size"] * 8,
            num_workers=cfgs["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=inputs.get_collater("test"),
            sampler=test_sampler,
        )

        self.meta_dict = self.test_loader.dataset.meta_dict
        self.ims = self.cfgs["model"]["inputs"]["image_size"]

    def load_model(self):
        self.cfgs["save_dir"] = misc.set_save_dir(self.cfgs)

        import models

        model = models.get_model(self.cfgs)
        self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
        model = model.to(self.device)

        with open(os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb") as f:
            if self.cfgs_test["epoch"] is not None:
                best_epoch = self.cfgs_test["best_epoch"]
            else:
                tot_val_record = pickle.load(f)
                best_epoch = tot_val_record["best"]["epoch"]

            load_model_dir = os.path.join(
                self.cfgs["save_dir"], "epoch_{}.pt".format(best_epoch)
            )
            print("Load: ", load_model_dir)
            checkpoint = torch.load(load_model_dir)
            model.load_state_dict(checkpoint["model"], strict=True)

        return model

    def do_test(self, model=None):
        print("Doing Inference.. ")
        self.model = self.load_model()

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
                        bi_det_preds[:, -1] >= self.cfgs_test["prob_ths"]
                    ]

                    if len(bi_det_preds) == 0:  # No pred bbox
                        bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    else:
                        bi_dimy = self.meta_dict[bi_fp]["dim0"]
                        bi_dimx = self.meta_dict[bi_fp]["dim1"]
                        bi_det_preds[:, [0, 2]] *= bi_dimx / self.ims
                        bi_det_preds[:, [1, 3]] *= bi_dimy / self.ims
                        bi_det_preds[:, :4] = np.round(bi_det_preds[:, :4])

                    bi_cls_pred = torch.sigmoid(logits["aux_cls"][bi]).item()
                    if self.cfgs["model"]["test"]["use_classifier"]:
                        if bi_cls_pred < self.cfgs["model"]["test"]["cls_th"]:
                            bi_det_preds = np.array([[0, 0, 1, 1, 14, 1]])

                    # TODO: to string format
                    pred_string = ""
                    for det_i in bi_det_preds:
                        x0, y0, x1, y1, c, cf = det_i
                        pred_string += (
                            f" {int(c)} {cf} {int(x0)} {int(y0)} {int(x1)} {int(y1)}"
                        )

                    pred_string = pred_string[1:]  # remove left blank

                    submit_df.append([bi_fp, pred_string])

        # TODO: make submit csv
        submit_csv = pd.DataFrame(submit_df, columns=["image_id", "PredictionString"])
        submit_dir = os.path.join(
            self.cfgs_test["submit_dir"],
            f"fold{self.cfgs['fold']}{self.cfgs['memo']}_submit.csv",
        )
        submit_csv.to_csv(submit_dir, index=False)
        print("Submission csv saved in: ", submit_dir)

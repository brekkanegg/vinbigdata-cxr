import os, sys
import torch
import numpy as np
import pandas as pd
import pickle
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils
from utils import misc
from inputs import vin
import pprint
import models
import random
import time


class Testor(object):
    def __init__(self, cfgs, device=None):

        self.cfgs = cfgs

        # modify for faster
        # self.cfgs["meta"]["inputs"]["nms_fn"] = "naive"
        self.cfgs["meta"]["model"]["nms_fn"] = "naive"

        self.cfgs["save_dir"] = misc.set_save_dir(cfgs)
        print(f"\n\nConfigs: \n{self.cfgs}\n")

        self.cfgs_test = self.cfgs["meta"]["test"]

        self.device = device

        ####### DATA
        self.test_loader = vin.get_dataloader(self.cfgs, mode="test")

        self.meta_dict = self.test_loader.dataset.meta_dict

    def load_model(self, load_dir):
        # self.cfgs["save_dir"] = misc.set_save_dir(self.cfgs)

        model = models.get_model(self.cfgs, pretrained=False)
        self.device = torch.device(f"cuda:{self.cfgs['local_rank']}")
        model = model.to(self.device)

        with open(load_dir + "/tot_val_record.pkl", "rb") as f:
            tot_val_record = pickle.load(f)

        best_epoch = self.cfgs["meta"]["test"]["best_epoch"]
        if best_epoch is None:
            best_epoch = tot_val_record["best"]["epoch"]

        load_model_dir = os.path.join(load_dir, f"epoch_{best_epoch}.pt")

        print("Load: ", load_model_dir)
        pprint.pprint(tot_val_record[str(best_epoch)])

        checkpoint = torch.load(load_model_dir)
        model.load_state_dict(checkpoint["model"], strict=True)

        return model

    def do_test(self):

        print("Inference Detector, But Use only Auxiliary Output!")

        if self.cfgs_test["submit_name"] is None:
            try:
                submit_name = input("Submit csv name: ")
            except SyntaxError:
                print("Enter submit csv name")
                raise ()
        else:
            submit_name = self.cfgs_test["submit_name"]

        print("Doing Inference.. ")
        self.model = self.load_model(self.cfgs["save_dir"])

        submit_df = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():
            t0 = time.time()
            for data in tqdm(self.test_loader):
                # t1 = time.time()
                # print("load data:", t1 - t0)
                img = data["img"].permute(0, 3, 1, 2).to(self.device)
                logits = self.model(img, mode="test_only_class")
                # t2 = time.time()
                # print("inference:", t2 - t1)

                for bi in range(len(data["fp"])):
                    bi_fp = data["fp"][bi]
                    bi_cls_pred = torch.sigmoid(logits["aux_cls"][bi]).item()
                    submit_df.append([bi_fp, bi_cls_pred])
                # t3 = time.time()
                # print("etc:", t3 - t2)
                # t0 = t3

        # Make submit csv
        submit_csv = pd.DataFrame(submit_df, columns=["image_id", "is_abnormal"])

        submit_dir = os.path.join(
            self.cfgs_test["submit_dir"], f"{submit_name}_class.csv"
        )
        submit_csv.to_csv(submit_dir, index=False)
        print("Submission csv saved in: ", submit_dir)

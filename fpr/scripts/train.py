import os, sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import pickle
import datetime

from scripts.validate import Validator

import utils
from utils import misc
from inputs import patch

import timm

import opts


class Trainer(object):
    def __init__(self, cfgs):

        ####### CONFIGS
        self.cfgs = cfgs
        self.cfgs["save_dir"] = misc.set_save_dir(cfgs)
        os.makedirs(self.cfgs["save_dir"], exist_ok=True)

        ####### Logging
        self.tb_writer = utils.get_writer(self.cfgs)
        self.txt_logger = utils.get_logger(self.cfgs)

        self.txt_logger.write("\n\n----train fpr.py----")
        self.txt_logger.write(f"\n{datetime.datetime.now()}")
        self.txt_logger.write(f"\n\nSave Directory: \n{self.cfgs['save_dir']}")
        self.txt_logger.write(f"\n\nConfigs: \n{self.cfgs}\n")

        ####### MODEL

        self.device = torch.device(f"cuda:{self.cfgs['local_rank']}")

        mname = self.cfgs["meta"]["model"]["name"]
        pretrained = not self.cfgs["meta"]["train"]["resume_train"]

        model = timm.create_model(
            f"tf_efficientnet_{mname}_ns", pretrained=pretrained, num_classes=1
        )

        self.model = model.to(self.device)

        ####### Data
        self.train_loader = patch.get_dataloader(self.cfgs, mode="train")
        self.txt_logger.write(f"\nTrain:  \n{len(self.train_loader.dataset)}\n")

        ####### Opts
        self.optimizer = opts.get_optimizer(self.cfgs, self.model.parameters())
        self.scheduler = opts.get_scheduler(self.cfgs, self.optimizer)
        self.grad_scaler = GradScaler(enabled=self.cfgs["use_amp"])

        self.criterion = nn.BCEWithLogitsLoss()

        ####### Validator
        self.validator = Validator(self.cfgs, self.device)
        self.txt_logger.write(f"Val:  \n{len(self.validator.val_loader.dataset)}\n")

    def do_train(self):

        ####### Setup Train
        self.epoch, self.iter, self.resume_epoch = 0, 0, 0
        self.tot_val_record = {
            "best": {"loss": np.inf, "auroc": -1, "recall": -1, "precision": -1}
        }

        if self.cfgs["meta"]["train"]["resume_train"]:
            with open(
                os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb"
            ) as f:
                self.tot_val_record = pickle.load(f)
                self.iter, self.resume_epoch = (
                    self.tot_val_record["best"]["iteration"],
                    self.tot_val_record["best"]["epoch"],
                )
                resume_model_dir = os.path.join(
                    self.cfgs["save_dir"], f"epoch_{self.resume_epoch}.pt"
                )
                checkpoint = torch.load(resume_model_dir)
                self.model.load_state_dict(checkpoint["model"], strict=True)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.grad_scaler.load_state_dict(checkpoint["scaler"])
                self.txt_logger.write("\n\nResume Training Here! \n\n")

        self.txt_logger.write("\n\nStart Training! \n\n")

        ####### Train
        self.start_time = time.time()
        self.endurance = 0
        for epoch in range(self.resume_epoch, self.cfgs["meta"]["train"]["max_epoch"]):

            self.one_epoch_steps = len(self.train_loader)
            if self.cfgs["memo"] == "dryrun":
                self.display_step = 1
            else:
                self.display_step = (
                    self.one_epoch_steps
                    // self.cfgs["meta"]["train"]["display_interval"]
                )

            self.epoch = epoch
            if self.endurance > self.cfgs["meta"]["train"]["endurance"]:
                self.txt_logger.write(
                    "\nStop training! No more performance gain expected!"
                )
                best_epoch = self.tot_val_record["best"]["epoch"]
                self.txt_logger.write(
                    f"\n\nBest saved at: {self.cfgs['save_dir']}, {best_epoch} epoch\n\n"
                )
                break
            self.train_val_one_epoch()

    def train_val_one_epoch(self):

        self.optimizer.zero_grad()
        self.model.train()

        nums_tot = 0
        losses_tot = 0

        # Shuffle
        # TODO: normal - abnormal 1:1
        if self.cfgs["meta"]["train"]["posneg_ratio"] == 1:
            random.shuffle(self.train_loader.dataset.fps)
            random.shuffle(self.train_loader.dataset.gts)
        else:
            random.shuffle(self.train_loader.dataset.tot_files)

        for i, data in enumerate(self.train_loader):

            img = data["img"].permute(0, 3, 1, 2).to(self.device)
            logits = self.model(img)

            loss = self.criterion(logits, data["label"].to(self.device))

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            losses_tot += loss * len(data["fp"])
            nums_tot += len(data["fp"])

            take_time = utils.convert_time(time.time() - self.start_time)

            self.txt_logger.write(
                f"\repoch-step: {self.epoch}-{i}/{self.one_epoch_steps}, time: {take_time}, loss: {loss:.4f}",
                False,
            )

            self.iter += 1

            lr0 = self.cfgs["meta"]["opts"]["learning_rate"]
            wep = self.cfgs["meta"]["opts"]["warmup_epoch"]
            if self.epoch < wep:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr0 / wep * (self.epoch + i / self.one_epoch_steps)
            else:
                if not self.scheduler is None:
                    self.scheduler.step(self.epoch - wep + i / self.one_epoch_steps)

        avg_loss = losses_tot / (nums_tot + 1e-5)
        self.tb_writer.write_scalars({"loss": {"t loss": avg_loss}}, self.epoch)

        # Do Validation
        do_val = self.epoch > self.cfgs["meta"]["val"]["ignore_epoch"]
        if do_val:
            val_record = self.validator.do_validate(self.model)
            self.tot_val_record[str(self.epoch + 1)] = val_record
            val_best = val_record[self.cfgs["meta"]["val"]["best"]]

            # Save Model
            select_metric = self.cfgs["meta"]["val"]["best"]
            val_improved = False
            if select_metric == "auroc":
                if val_best >= self.tot_val_record["best"][select_metric]:
                    val_improved = True

            elif select_metric == "loss":
                if val_best < self.tot_val_record["best"][select_metric]:
                    val_improved = True

            if val_improved:
                checkpoint = {
                    "epoch": self.epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.grad_scaler.state_dict(),
                }
                model_name = os.path.join(
                    self.cfgs["save_dir"], "epoch_" + str(self.epoch + 1) + ".pt"
                )
                torch.save(checkpoint, model_name)
                self.tot_val_record["best"] = val_record
                self.tot_val_record["best"]["epoch"] = self.epoch + 1
                self.tot_val_record["best"]["iteration"] = self.iter
                self.endurance = 0
            else:
                self.endurance += 1

            take_time = utils.convert_time(time.time() - self.start_time)
            vloss = val_record["loss"]
            vbest_epoch = self.tot_val_record["best"]["epoch"]

            self.txt_logger.write(
                f"\repoch: {self.epoch+1}, time: {take_time}, tloss: {avg_loss:.4f}, vloss: {vloss:.4f}"
            )
            self.txt_logger.write("\n")

            for k in ["auroc", "recall", "precision", "threshold"]:
                self.txt_logger.write(f"{k}: {val_record[k]}  ")
            self.txt_logger.write("\n")
            self.txt_logger.write(
                f"best epoch: {vbest_epoch} / {self.tot_val_record['best'][select_metric]:.4f}"
            )
            self.txt_logger.write("\n", txt_write=True)
            self.txt_logger.write("\n", txt_write=False)

            # metric_keys = ["auroc", "recall", "precision"]
            # self.tb_writer.write_scalars(
            #     {"metrics": {f"{key}": val_record[key] for key in metric_keys}},
            #     self.epoch,
            # )

            self.tb_writer.write_scalars({"loss": {"v loss": vloss}}, self.epoch)

            with open(
                os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "wb"
            ) as f:
                pickle.dump(self.tot_val_record, f)

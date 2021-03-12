import os, sys
import time
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
import pickle
import datetime

from tqdm import tqdm
from pathlib import Path
from scripts.validate import Validator

import utils
import inputs

import models
import opts


class Trainer(object):
    def __init__(self, cfgs):

        save_dict = OrderedDict()

        save_dict["fold"] = cfgs["fold"]
        if cfgs["memo"] is not None:
            save_dict["memo"] = cfgs["memo"]  # 1,2,3
        specific_dir = ["{}-{}".format(key, save_dict[key]) for key in save_dict.keys()]

        cfgs["save_dir"] = os.path.join(
            cfgs["save_dir"],
            "_".join(specific_dir),
        )
        os.makedirs(cfgs["save_dir"], exist_ok=True)

        ####### CONFIGS
        self.cfgs = cfgs

        ####### Logging
        self.tb_writer = utils.get_writer(self.cfgs)
        self.txt_logger = utils.get_logger(self.cfgs)

        self.txt_logger.write("\n\n----train.py----")
        self.txt_logger.write("\n{}".format(datetime.datetime.now()))
        self.txt_logger.write("\n\nSave Directory: \n{}".format(self.cfgs["save_dir"]))
        self.txt_logger.write("\n\nConfigs: \n{}\n".format(self.cfgs))

        ####### MODEL
        model = models.get_model(self.cfgs)
        self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
        self.model = model.to(self.device)

        ####### Data

        train_dataset = inputs.get_dataset(self.cfgs, mode="train")
        train_sampler = None
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfgs["batch_size"],
            num_workers=self.cfgs["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=inputs.get_collater(),
            sampler=train_sampler,
        )

        ####### Opts
        self.optimizer = opts.get_optimizer(self.cfgs, self.model.parameters())
        self.scheduler = opts.get_scheduler(self.cfgs, self.optimizer)
        self.grad_scaler = GradScaler(enabled=self.cfgs["use_amp"])

        ####### Validator
        self.validator = Validator(self.cfgs, self.device)

    def do_train(self):

        ####### Setup Train
        self.epoch, self.iter, self.resume_epoch = 0, 0, 0
        self.tot_val_record = {"best": {"loss": np.inf, "coco": -1}}

        if self.cfgs["model"]["train"]["resume_train"]:
            with open(
                os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb"
            ) as f:
                self.tot_val_record = pickle.load(f)
                self.iter, self.resume_epoch = (
                    self.tot_val_record["best"]["iteration"],
                    self.tot_val_record["best"]["epoch"],
                )
                resume_model_dir = os.path.join(
                    self.cfgs["save_dir"], "epoch_{}.pt".format(self.resume_epoch)
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
        for epoch in range(self.resume_epoch, self.cfgs["model"]["train"]["max_epoch"]):

            self.one_epoch_steps = len(self.train_loader)
            self.display_step = (
                self.one_epoch_steps // self.cfgs["model"]["train"]["display_interval"]
            )

            self.epoch = epoch
            if self.endurance > self.cfgs["model"]["train"]["endurance"]:
                self.txt_logger.write(
                    "\nStop training! No more performance gain expected!"
                )
                best_epoch = self.tot_val_record["best"]["epoch"]
                self.txt_logger.write(
                    "\n\nBest saved at: {}, {} epoch\n\n".format(
                        self.cfgs["save_dir"], best_epoch
                    )
                )
                break
            self.train_val_one_epoch()

    def train_val_one_epoch(self):

        self.optimizer.zero_grad()
        self.model.train()

        for i, data in enumerate(self.train_loader):

            img = data["img"].permute(0, 3, 1, 2).to(self.device)
            logits = self.model(img)

            dloss, closs = opts.calc_loss(self.cfgs, self.device, data, logits)
            loss = dloss + self.cfgs["model"]["loss"]["cls_weight"] * closs

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            loss = loss.detach().item()
            dloss = dloss.detach().item()
            closs = closs.detach().item()

            take_time = utils.convert_time(time.time() - self.start_time)

            self.txt_logger.write(
                f"\repoch-step: {self.epoch}-{i}/{self.one_epoch_steps}, time: {take_time}, dloss: {dloss:.4f}, closs: {closs:.4f}",
                False,
            )

            self.tb_writer.write_scalars({"loss": {"t loss": loss}}, self.iter)
            self.tb_writer.write_scalars({"dloss": {"t dloss": dloss}}, self.iter)
            self.tb_writer.write_scalars({"closs": {"t closs": closs}}, self.iter)

            if self.iter % self.display_step == 0:
                for viz_bi in range(len(data["fp"])):
                    if data["bbox"][viz_bi, 0, -1] != -1:
                        break

                with torch.no_grad():
                    self.model.eval()
                    det_preds_viz = (
                        self.model(img, mode="viz")["preds"][viz_bi]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if len(det_preds_viz) == 0:
                        det_preds_viz = np.ones((1, 6)) * -1

                    det_anns_viz = data["bbox"][viz_bi].numpy()

                    self.tb_writer.write_images(
                        data["fp"][viz_bi],
                        data["img"][viz_bi].numpy(),
                        det_preds_viz,
                        det_anns_viz,
                        self.iter,
                        "train",
                    )
                    self.model.train()

            self.iter += 1

            lr0 = self.cfgs["model"]["opts"]["learning_rate"]
            wep = self.cfgs["model"]["opts"]["warmup_epoch"]
            if self.epoch < wep:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr0 / wep * (self.epoch + i / self.one_epoch_steps)
            else:
                if not self.scheduler is None:
                    self.scheduler.step(self.epoch - wep + i / self.one_epoch_steps)

        if self.epoch > self.cfgs["model"]["val"]["ignore_epoch"]:

            # Do Validation
            val_record, val_viz = self.validator.do_validate(self.model)
            self.tot_val_record[str(self.epoch + 1)] = val_record
            val_best = val_record[self.cfgs["model"]["val"]["best"]]

            # Save Model
            select_metric = self.cfgs["model"]["val"]["best"]
            val_improved = False
            if select_metric == "loss":
                if val_best < self.tot_val_record["best"][select_metric]:
                    val_improved = True
            elif select_metric == "coco":
                if val_best > self.tot_val_record["best"][select_metric]:
                    val_improved = True

            # if val_improved:
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
            # else:
            #     self.endurance += 1

            take_time = utils.convert_time(time.time() - self.start_time)
            vloss = val_record["loss"]
            vdloss = val_record["dloss"]
            vcloss = val_record["closs"]
            vbest_epoch = self.tot_val_record["best"]["epoch"]
            # metric_keys = ["det_recl", "det_prec", "det_fppi", "det_f1"]
            # metric_keys += ["cls_auc", "cls_sens", "cls_spec"]
            # val_logs = [vloss] + [val_record[k] for k in metric_keys]

            self.txt_logger.write(
                f"\repoch: {self.epoch+1}, time: {take_time}, tdloss: {dloss:.4f}, tcloss: {closs:.4f}, vdloss: {vdloss:.4f}, vcloss: {vcloss:.4f}"
            )
            self.txt_logger.write("\n")
            for k in ["det_recl", "det_prec", "det_fppi", "det_f1"]:
                self.txt_logger.write(f"{k}: ")
                for v in val_record[k]:
                    self.txt_logger.write(f"{v:.2f} ")
                self.txt_logger.write("\n")
            for k in ["cls_auc", "cls_sens", "cls_spec", "coco"]:
                self.txt_logger.write(f"{k}: {val_record[k]:.2f}")
                self.txt_logger.write("\n")
            self.txt_logger.write(f"best epoch: {vbest_epoch}")
            self.txt_logger.write("\n", txt_write=True)
            self.txt_logger.write("\n", txt_write=False)

            self.tb_writer.write_images(
                val_viz["fp"],
                val_viz["img"],
                val_viz["pred"],
                val_viz["ann"],
                self.iter,
                "val",
            )

            self.tb_writer.add_scalar("coco", val_record["coco"], self.epoch)
            metric_keys = ["det_recl", "det_prec", "det_fppi", "det_f1"]
            self.tb_writer.write_scalars(
                {
                    "metrics": {
                        "{}".format(key): val_record[key][:-1].mean()
                        for key in metric_keys
                    }
                },
                self.iter,
            )

            metric_keys = ["cls_auc", "cls_sens", "cls_spec"]
            self.tb_writer.write_scalars(
                {"metrics": {"{}".format(key): val_record[key] for key in metric_keys}},
                self.iter,
            )

            self.tb_writer.write_scalars({"loss": {"v loss": vloss}}, self.iter)
            self.tb_writer.write_scalars({"dloss": {"v dloss": vdloss}}, self.iter)
            self.tb_writer.write_scalars({"closs": {"v closs": vcloss}}, self.iter)

            with open(
                os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "wb"
            ) as f:
                pickle.dump(self.tot_val_record, f)

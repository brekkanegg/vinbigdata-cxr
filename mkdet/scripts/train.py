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

from utils import tools
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
            # cfgs["model"]["meta"],
            # cfgs["model"]["inputs"]["label"],
            "_".join(specific_dir),
        )
        os.makedirs(cfgs["save_dir"], exist_ok=True)

        ####### CONFIGS
        self.cfgs = cfgs

        ####### Logging
        self.tb_writer = utils.get_writer(self.cfgs)
        self.txt_logger = utils.get_logger(self.cfgs)

        self.do_logging = True
        if len(self.cfgs["gpu"]) > 1:
            if dist.get_rank() != 0:
                self.do_logging = False

        if self.do_logging:
            self.txt_logger.write("\n\n----train.py----")
            self.txt_logger.write("\n{}".format(datetime.datetime.now()))
            self.txt_logger.write(
                "\n\nSave Directory: \n{}".format(self.cfgs["save_dir"])
            )
            self.txt_logger.write("\n\nConfigs: \n{}\n".format(self.cfgs))

        ####### MODEL
        model = models.get_model(self.cfgs)
        if len(self.cfgs["gpu"]) > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
            self.model = model.to(self.device)
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.cfgs["local_rank"]],
                output_device=self.cfgs["local_rank"],
            )
        else:
            self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
            self.model = model.to(self.device)

        ####### Data

        train_dataset = inputs.get_dataset(self.cfgs, mode="train")
        if len(self.cfgs["gpu"]) > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=len(self.cfgs["gpu"]),
                rank=self.cfgs["local_rank"],
            )
        else:
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

        # if self.do_logging:
        #     self.txt_logger.write("\nDataset: ")
        #     self.txt_logger.write(
        #         "\nTRAIN Abnormal/Normal: {}/{}".format(
        #             len(train_dataset.abnormal_meta_df),
        #             len(train_dataset.normal_meta_df),
        #         )
        #     )

        ####### Opts
        self.optimizer = opts.get_optimizer(self.cfgs, self.model.parameters())
        self.scheduler = opts.get_scheduler(self.cfgs, self.optimizer)
        self.grad_scaler = GradScaler(enabled=self.cfgs["use_amp"])

        ####### Validator
        self.validator = Validator(self.cfgs, self.device)
        # if self.do_logging:
        #     self.txt_logger.write(
        #         "\nVAL   Abnormal/Normal: {}/{}".format(
        #             len(self.validator.val_loader.dataset.abnormal_meta_df),
        #             len(self.validator.val_loader.dataset.normal_meta_df),
        #         )
        #     )

        # if self.cfgs["model"]["val"]["ignore_normal"]:
        #     self.txt_logger.write("\nVAL   Ignore Normal")
        #     self.validator.val_loader.dataset.meta_df = (
        #         self.validator.val_loader.dataset.abnormal_meta_df
        #     )

    def do_train(self):

        ####### Setup Train
        self.epoch, self.iter, self.resume_epoch = 0, 0, 0
        self.tot_val_record = {
            "best": {"det_recl": -1, "det_prec": -1, "det_f1": -1, "loss": np.inf}
        }

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

        if self.do_logging:
            self.txt_logger.write("\n\nStart Training! \n\n")
            header_columns = ["epoch", "iter", "time", "train_loss", "val_loss"]
            header_columns += ["det_recl", "det_prec", "det_fppi", "det_f1"]
            header_columns += ["cls_auc", "cls_sens", "cls_spec"]
            header_columns += ["best_epoch"]
            self.txt_logger.log_header(header_columns)

        ####### Train
        self.start_time = time.time()
        self.endurance = 0
        for epoch in range(self.resume_epoch, self.cfgs["model"]["train"]["max_epoch"]):
            # self.train_loader.dataset.shuffle()
            # self.train_loader.dataset.meta_df = (
            #     self.train_loader.dataset.abnormal_meta_df
            # )

            self.one_epoch_steps = len(self.train_loader)
            self.display_step = (
                self.one_epoch_steps // self.cfgs["model"]["train"]["display_interval"]
            )

            self.epoch = epoch
            if self.endurance > self.cfgs["model"]["train"]["endurance"]:
                if self.do_logging:
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

        t0 = time.time()

        for i, data in enumerate(self.train_loader):
            t1 = time.time()
            img = data["img"].permute(0, 3, 1, 2).to(self.device)
            logit = self.model(img)

            t2 = time.time()

            # FIXME: GPU Util이 안 나온다
            loss = opts.calc_loss(self.cfgs, self.device, data, logit)

            t3 = time.time()

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()

            t4 = time.time()

            # NOTE: Try to avoid excessive CPU-GPU synchronization (.item() calls, or printing values from CUDA tensors).

            if self.do_logging:
                loss = loss.detach().item()
                take_time = tools.convert_time(time.time() - self.start_time)
                train_logs = [loss, "-"]
                self.txt_logger.log_result(
                    [self.epoch, "{}/{}".format(i, self.one_epoch_steps), take_time]
                    + train_logs
                )
                self.tb_writer.write_scalars(
                    {"loss": {"train loss": loss}},
                    self.iter,
                )

                if self.iter % self.display_step == 0:
                    # Visualize
                    # Find abnormal
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

                        if len(det_preds_viz) != 0:
                            # sigmoid
                            det_preds_viz[:, -1] = 1 / (
                                1 + np.exp(-1 * det_preds_viz[:, -1])
                            )
                        else:
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

            t5 = time.time()
            if self.cfgs["do_profiling"]:
                print("\ndata", t1 - t0)
                print("forward", t2 - t1)
                print("calc loss", t3 - t2)
                print("backward", t4 - t3)
                print("logging", t5 - t4)
            t0 = t5

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
            elif select_metric == "det_f1":
                if val_best > self.tot_val_record["best"][select_metric]:
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

            if self.do_logging:
                take_time = utils.tools.convert_time(time.time() - self.start_time)
                vloss = val_record["loss"]
                vbest_epoch = self.tot_val_record["best"]["epoch"]
                metric_keys = ["det_recl", "det_prec", "det_fppi", "det_f1"]
                metric_keys += ["cls_auc", "cls_sens", "cls_spec"]
                val_logs = [vloss] + [val_record[k] for k in metric_keys]
                self.txt_logger.log_result(
                    [self.epoch + 1, self.iter, take_time, loss]
                    + val_logs
                    + [vbest_epoch],
                    txt_write=True,
                )
                self.txt_logger.write("\n", txt_write=True)
                self.tb_writer.write_images(
                    val_viz["fp"],
                    val_viz["img"],
                    val_viz["pred"],
                    val_viz["ann"],
                    self.iter,
                    "val",
                )

                self.tb_writer.write_scalars(
                    {
                        "metrics": {
                            "{}".format(key): val_record[key] for key in metric_keys
                        }
                    },
                    self.iter,
                )
                self.tb_writer.write_scalars({"loss": {"val loss": vloss}}, self.iter)

                with open(
                    os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "wb"
                ) as f:
                    pickle.dump(self.tot_val_record, f)

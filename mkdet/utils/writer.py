import os
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2


import warnings

warnings.filterwarnings("ignore")


class Writer(SummaryWriter):
    def __init__(self, cfgs):
        super().__init__(log_dir=cfgs["save_dir"])
        self.cfgs = cfgs

    def write_scalar(self, scalar_dict, iteration):
        for k, v in scalar_dict.items():
            self.add_scalar(k, v, iteration)

    def write_scalars(self, scalars_dict, iteration):
        for k1, v1 in scalars_dict.items():
            self.add_scalars(k1, {k2: v2 for k2, v2 in v1.items()}, iteration)

    def write_images(self, fp, img, pred, gt, iteration=None, mode="train"):
        # NOTE:  pred, gt shape: [num_class, size, size]

        if mode == "test":
            th = self.cfgs["model"]["test"]["prob_ths"][0]
            save_mode = self.cfgs["model"]["test"]["save_mode"]
            save_only_wrong = self.cfgs["model"]["test"]["save_only_wrong"]

        else:
            th = self.cfgs["model"]["val"]["prob_ths"][0]
            save_mode = ""
            save_only_wrong = False

        # num_classes = pred.shape[0]
        # pred = (pred > th).astype(np.float64)
        # gt = (gt > 0.5).astype(np.float64)  # For soft-label case

        gt_bbox, gt_class = gt[:, :4], gt[:, 4]

        pred_bbox, pred_class, pred_score = pred[:, :4], pred[:, 4], pred[:, 5]
        pred_idx = pred_score > th
        pred_score = pred_score[pred_idx]
        pred_class = pred_class[pred_idx]
        pred_bbox = pred_bbox[pred_idx]

        csv_name = fp.split("png_1024/")[1].split("/")[0]
        fp = fp.split("png_1024/")[1].replace("/", "-")
        img_dir_name = "images" + save_mode
        if save_only_wrong:
            img_dir_name += "_wrong"

        fig = plt.figure()
        fig.suptitle(fp)
        plt.axis("off")

        ax00 = fig.add_subplot(1, 3, 1)
        ax00.imshow(img, cmap="gray")

        ax01 = fig.add_subplot(1, 3, 2)
        ax01.imshow(img, cmap="gray")
        for i, (i_gtb, i_gtc) in enumerate(zip(gt_bbox, gt_class)):
            if i_gtc == -1:
                continue
            x0, y0, x1, y1 = [int(ii) for ii in i_gtb]
            w, h = (x1 - x0), (y1 - y0)
            rect = patches.Rectangle(
                (x0, y0), w, h, linewidth=1, edgecolor="g", facecolor="none"
            )
            # plt.text(
            #     x0,
            #     y0,
            #     int(i_gtc + 1),
            #     bbox={"facecolor": "g", "alpha": 0.5, "pad": 0},
            # )
            ax01.add_patch(rect)

        # 겹치면- 분홍, seg만- 보라, pred만- 빨강
        ax02 = fig.add_subplot(1, 3, 3)
        ax02.imshow(img, cmap="gray")
        for i, (i_pb, i_pc, i_ps) in enumerate(zip(pred_bbox, pred_class, pred_score)):
            if i_pc == -1:
                continue

            x0, y0, x1, y1 = [int(ii) for ii in i_pb]
            w, h = (x1 - x0), (y1 - y0)
            rect = patches.Rectangle(
                (x0, y0), w, h, linewidth=1, edgecolor="r", facecolor="none"
            )
            # plt.text(
            #     x0,
            #     y0,
            #     int(i_pc + 1),
            #     bbox={"facecolor": "r", "alpha": 0.5, "pad": 0},
            # )
            # i_ps = "{:.2f}".format(i_ps)
            # plt.text(x0, y0, i_ps, bbox={"facecolor": "r", "alpha": 0.5, "pad": 0})
            ax02.add_patch(rect)

        if save_mode == "concat":
            plt_dir = (
                os.path.join(self.logdir, img_dir_name, csv_name, fp[:-4]) + "_plt.png"
            )
            os.makedirs(os.path.dirname(plt_dir), exist_ok=True)
            plt.savefig(plt_dir)
            plt.close(fig)

        else:
            self.add_figure("{}_img".format(mode), fig, iteration, close=True)
            plt.close(fig)
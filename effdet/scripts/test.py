import os, sys

import torch
import numpy as np
import pickle
import cv2
from collections import OrderedDict

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import datetime
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import opts
import utils
import inputs
import models
from metrics.metrics import evaluate

# TODO: Make Inference.py for no label-case


class Testor(object):
    def __init__(self, cfgs):

        save_dict = OrderedDict()
        save_dict["fold"] = cfgs["fold"]
        if cfgs["memo"] is not None:
            save_dict["memo"] = cfgs["memo"]  # 1,2,3

        specific_dir = ["{}-{}".format(key, save_dict[key]) for key in save_dict.keys()]

        cfgs["save_dir"] = os.path.join(
            cfgs["save_dir"],
            cfgs["model"]["meta"],
            cfgs["model"]["inputs"]["label"],
            "_".join(specific_dir),
        )

        # cfgs["save_dir"] = os.path.join(cfgs["save_dir"], "_".join(specific_dir))
        os.makedirs(cfgs["save_dir"], exist_ok=True)

        self.cfgs = cfgs
        self.cfgs_test = cfgs["model"]["test"]

        self.tb_writer = utils.get_writer(self.cfgs)
        self.txt_logger = utils.get_logger(self.cfgs)

        self.txt_logger.write("\n\n----test.py----")
        self.txt_logger.write("\n{}".format(datetime.datetime.now()))
        self.txt_logger.write("\n\nSave Directory: \n{}".format(self.cfgs["save_dir"]))
        self.txt_logger.write("\n\nConfigs: \n{}\n".format(self.cfgs))

        ####### MODEL
        # NOTE: No Multiple GPU Support for Test
        model = models.get_model(self.cfgs)
        self.device = torch.device("cuda:{}".format(self.cfgs["local_rank"]))
        self.model = model.to(self.device)

    def do_test(self):

        with open(os.path.join(self.cfgs["save_dir"], "tot_val_record.pkl"), "rb") as f:
            tot_val_record = pickle.load(f)
            best_record = tot_val_record["best"]
            best_epoch = best_record["epoch"]
            best_model_dir = os.path.join(
                self.cfgs["save_dir"], "epoch_{}.pt".format(best_epoch)
            )
            checkpoint = torch.load(best_model_dir)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self.txt_logger.write("\n Load model dir: {}".format(best_model_dir))
            self.txt_logger.write(
                "\n" + str({k: v for k, v in best_record.items() if k != "viz"})
            )

        self.test_csvs = inputs.get_dataset(self.cfgs, mode="test").csv_files
        self.txt_logger.write("\n\nCSVS: " + str(self.test_csvs))

        header_columns = ["# pos", "# neg", "# lesions", "# tp", "# fp"]
        header_columns += ["les sens", "les prec", "les fppi", "les f1"]
        header_columns += ["img AUC", "img sens", "img spec"]

        self.txt_logger.write("\n\n")
        self.txt_logger.log_header(header_columns)

        for specific_csv in self.test_csvs:
            self.test_specific_csv(specific_csv)

    def genActivationMapGrayScale(self, output, saveimg, min_threshold=0.0001):
        dtype = np.uint8
        _, thr = cv2.threshold(output, min_threshold, 255, 0)
        thr = thr.astype(dtype)
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        font_size = 0.0005 * (output.shape[0] + output.shape[1]) / 2
        thick_size = int(9 * font_size)
        thick_size_2 = int(thick_size / 2)
        # print(font_size, thick_size)
        for cnt in contours:
            M = cv2.moments(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            max_p = -1
            for i in range(x, x + w):
                for j in range(y, y + h):
                    if max_p < output[j][i]:
                        max_p = output[j][i]
            cX = int(M["m10"] / (M["m00"] + 1e-4))
            cY = int(M["m01"] / (M["m00"] + 1e-4))
            cv2.drawContours(saveimg, [cnt], 0, (255, 255, 255, 255), thick_size)
            cv2.drawContours(saveimg, [cnt], 0, (0, 0, 0, 255), thick_size_2)
            text_x = int(cX - 30 * font_size)
            text_y = int(cY + h / 2 + 45 * font_size)
            cv2.putText(
                saveimg,
                str(int(round(100 * max_p))) + "%",
                (text_x, text_y),
                0,
                font_size,
                (255, 255, 255, 255),
                thick_size,
            )
            cv2.putText(
                saveimg,
                str(int(round(100 * max_p))) + "%",
                (text_x, text_y),
                0,
                font_size,
                (0, 0, 0, 255),
                thick_size_2,
            )

        return saveimg

    # FIXME:
    def save_image(self, fp, img, pred, gt, wrong_type="fp"):

        num_classes = pred.shape[0]
        pred[pred < 0.1] = 0
        gt = (gt > 0.5).astype(np.uint8)  # For soft-label case

        csv_name = fp.split("png_1024/")[1].split("/")[0]
        fp = fp.split("png_1024/")[1].replace("/", "-")
        img_dir_name = "images_" + wrong_type

        if img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024))
            pred_temp = np.zeros((num_classes, 1024, 1024))
            gt_temp = np.zeros((num_classes, 1024, 1024))
            for c in range(num_classes):
                pred_temp[c, :, :] = cv2.resize(pred[c, :, :], (1024, 1024))
                gt_temp[c, :, :] = cv2.resize(gt[c, :, :], (1024, 1024))
            pred = pred_temp
            gt = gt_temp

        # png_dir = os.path.join(self.cfgs["save_dir"], img_dir_name, fp)
        png_dir = os.path.join(self.cfgs["save_dir"], csv_name, img_dir_name, fp)
        os.makedirs(os.path.dirname(png_dir), exist_ok=True)

        img *= 255
        cv2.imwrite(png_dir, img)

        img_mod = np.dstack([img, np.ones((1024, 1024)) * 255.0])
        img_mod = img_mod.astype(np.uint8)

        for c in range(num_classes):
            cv2.imwrite(
                png_dir.replace(".png", "_{}_gt.png".format(c)), gt[c, :, :] * 255
            )

            pred_img = self.genActivationMapGrayScale(pred[c, :, :], img_mod)
            cv2.imwrite(
                png_dir.replace(".png", "_{}_pred.png".format(c)),
                pred_img,
            )

    def test_specific_csv(self, specific_csv):
        ####### DATA
        test_dataset = inputs.get_dataset(
            self.cfgs,
            mode="test",
            specific_csv=specific_csv,
        )

        # csv_name = specific_csv.split("/")[-1].split(".csv")[0]
        test_logs = [
            len(test_dataset.abnormal_meta_df),
            len(test_dataset.normal_meta_df),
        ]

        test_sampler = None
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfgs["batch_size"],
            num_workers=self.cfgs["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=inputs.get_collater(),
            sampler=test_sampler,
        )

        ####### Init test result
        nc = self.cfgs["model"]["inputs"]["num_classes"]
        npth = len(self.cfgs["model"]["test"]["prob_ths"])

        det_gt_nums_tot = 0  # np.zeros((npth, nc))
        det_tp_nums_tot = 0  # np.array([0] * nc)
        det_fp_nums_tot = 0  # np.array([0] * nc)
        det_pred_nums_tot = 0  # np.array([0] * nc)
        nums_tot = 0
        # losses_tot = 0

        cls_pred_tot = []
        cls_gt_tot = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():
            for data in tqdm(self.test_loader, disable=True):
                img = data["img"].permute(0, 3, 1, 2).to(self.device)
                logits = self.model(img, mode="test")
                # loss = opts.calc_loss(self.cfgs, self.device, data, logit)

                # losses_tot += loss * len(data["fp"])
                nums_tot += len(data["fp"])

                det_anns = data["bbox"].numpy()

                viz_bi = 0
                for bi in range(len(data["fp"])):
                    bi_det_preds = logits["preds"][bi].detach().cpu().numpy()
                    if len(bi_det_preds) == 0:  # No pred bbox
                        bi_det_preds = np.ones((1, 6)) * -1
                        bi_cls_pred = 0
                    else:
                        bi_det_preds[:, -1] = 1 / (1 + np.exp(-1 * bi_det_preds[:, -1]))
                        bi_cls_pred = np.max(bi_det_preds[:, -1])

                    bi_det_ann = det_anns[bi]
                    (
                        bi_det_gt_num,
                        bi_det_pred_num,
                        bi_det_tp_num,
                        bi_det_fp_num,
                    ) = evaluate(self.cfgs, bi_det_preds, bi_det_ann)

                    # FIXME: remove .item() if multi-label case
                    det_gt_nums_tot += bi_det_gt_num.item()
                    det_tp_nums_tot += bi_det_tp_num.item()
                    # correct_nums_tot += correct_num
                    det_pred_nums_tot += bi_det_pred_num.item()
                    det_fp_nums_tot += bi_det_fp_num.item()

                    cls_pred_tot.append(bi_cls_pred)
                    cls_gt_tot.append(int(bi_det_ann[0][-1] + 1))

                    # # for Visualization - abnormal

                    do_save = True
                    wrong_type = ""
                    if self.cfgs_test["save_only_wrong"]:
                        if bi_det_gt_num > 0:
                            if bi_det_tp_num == bi_det_gt_num:
                                do_save = False
                            else:
                                wrong_type = "fn"

                        else:
                            if bi_det_fp_num == 0:
                                do_save = False
                            else:
                                wrong_type = "fp"

                    if do_save:
                        nc = self.cfgs.model.inputs.num_classes
                        isz = self.cfgs.model.inputs.image_size

                        bi_det_preds_viz = logits["preds"][bi].detach().cpu().numpy()
                        if len(bi_det_preds_viz) != 0:  # No pred bbox
                            bi_det_preds_viz[:, -1] = 1 / (
                                1 + np.exp(-1 * bi_det_preds_viz[:, -1])
                            )
                        else:
                            bi_det_preds_viz = np.ones((1, 6)) * -1

                        bi_det_anns_viz = data["bbox"][bi].detach().cpu().numpy()

                        bi_det_preds_viz_temp = np.zeros((nc, isz, isz))
                        for (x0, y0, x1, y1, c, s) in bi_det_preds_viz:
                            if c == -1:
                                continue
                            else:
                                bi_det_preds_viz_temp[
                                    int(c), int(y0) : int(y1), int(x0) : int(x1)
                                ] = s

                        bi_det_anns_viz_temp = np.zeros((nc, isz, isz))
                        for (x0, y0, x1, y1, c) in bi_det_anns_viz:
                            if c == -1:
                                continue
                            else:
                                bi_det_anns_viz_temp[
                                    int(c), int(y0) : int(y1), int(x0) : int(x1)
                                ] = 1

                        self.save_image(
                            data["fp"][bi],
                            data["img"][bi].numpy(),
                            bi_det_preds_viz_temp,
                            bi_det_anns_viz_temp,
                            wrong_type,
                        )

        #####################################

        cls_pred_tot = np.array(cls_pred_tot)
        cls_gt_tot = np.array(cls_gt_tot)

        det_pc = det_tp_nums_tot / (det_pred_nums_tot + 1e-5)
        det_rc = det_tp_nums_tot / (det_gt_nums_tot + 1e-5)
        det_fppi = det_fp_nums_tot / nums_tot + 1e-5
        det_f1 = 2 * det_pc * det_rc / (det_pc + det_rc + 1e-5)

        try:
            cls_auc = roc_auc_score(cls_gt_tot, cls_pred_tot)
        except Exception as e:
            print(e)
            cls_auc = "None"

        if (cls_gt_tot == 0).all() and (cls_pred_tot == 0).all():
            cls_sens = "None"
            cls_spec = 1
        else:
            tn, fp, fn, tp = confusion_matrix(cls_gt_tot, cls_pred_tot > 0.5).ravel()
            cls_sens = tp / (tp + fn + 1e-5)
            cls_spec = tn / (tn + fp + 1e-5)

        test_logs += [str(det_gt_nums_tot), str(det_tp_nums_tot), str(det_fp_nums_tot)]
        test_logs += [det_rc, det_pc, det_fppi, det_f1]
        test_logs += [cls_auc, cls_sens, cls_spec]

        self.txt_logger.log_result(test_logs, txt_write=True)
        self.txt_logger.write("\n")

import os, sys
import torch
import numpy as np
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
import inputs
import opts
from metrics.metrics import evaluate


class Validator(object):
    def __init__(self, cfgs, device):

        self.cfgs = cfgs
        self.device = device

        ####### DATA
        val_transforms = None  # FIXME:
        val_dataset = inputs.get_dataset(self.cfgs, mode="val")
        if len(cfgs["gpu"]) > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=len(cfgs["gpu"]),
                rank=self.cfgs["local_rank"],
            )
        else:
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

        self.do_logging = True
        if len(self.cfgs["gpu"]) > 1:
            if dist.get_rank() != 0:
                self.do_logging = False

    def do_validate(self, model=None):

        self.model = model.to(self.device)

        ####### Init val result
        det_gt_nums_tot = np.zeros((15))
        det_tp_nums_tot = np.zeros((1, 15))
        det_fp_nums_tot = np.zeros((1, 15))
        det_pred_nums_tot = np.zeros((1, 15))
        nums_tot = 0
        losses_tot = 0

        # Classification metrics
        cls_pred_tot = []
        cls_gt_tot = []

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():

            # NOTE: For various visualization, no difference in dataset
            # if not self.cfgs["model"]["val"]["ignore_normal"]:
            # self.val_loader.dataset.shuffle()

            # self.val_loader.dataset.meta_df = self.val_loader.dataset.abnormal_meta_df

            tqdm_able = (self.cfgs["run"] != "train") and self.do_logging
            for data in tqdm(self.val_loader, disable=(not tqdm_able)):

                img = data["img"].permute(0, 3, 1, 2).to(self.device)

                logits = self.model(img, mode="val")
                loss = opts.calc_loss(self.cfgs, self.device, data, logits)

                if len(self.cfgs["gpu"]) > 1:
                    loss = torch.mean(gather_helper(loss))
                    data["fp"] = gather_helper(data["fp"])
                    data["bbox"] = gather_helper(data["bbox"])
                    # FIXME:
                    logits["preds"] = gather_helper(logits["preds"])
                    logits["aux_cls"] = gather_helper(logits["aux_cls"])

                loss = loss.detach().item()
                losses_tot += loss * len(data["fp"])
                nums_tot += len(data["fp"])

                det_anns = data["bbox"].numpy()

                # TODO: aux_classifier

                viz_bi = 0
                for bi in range(len(data["fp"])):
                    bi_det_preds = logits["preds"][bi].detach().cpu().numpy()
                    if len(bi_det_preds) == 0:  # No pred bbox
                        bi_det_preds = np.ones((1, 6)) * -1
                        bi_cls_pred = 0
                    else:
                        bi_det_preds[:, -1] = 1 / (1 + np.exp(-1 * bi_det_preds[:, -1]))
                        bi_cls_pred = np.max(bi_det_preds[:, -1])

                        # FIXME:
                        # bi_cls_pred = (
                        #     torch.sigmoid(logits["aux_cls"][bi])
                        #     .detach()
                        #     .cpu()
                        #     .numpy()[0]
                        # )

                        # if bi_cls_pred_max > 0.1:
                        #     bi_cls_pred_idx = np.argmax(bi_det_preds[:, -1], axis=-1)
                        #     bi_cls_pred = bi_det_preds[:, 4][bi_cls_pred_idx]
                        # else:
                        #     bi_cls_pred = 0

                    bi_det_ann = det_anns[bi]
                    (
                        bi_det_gt_num,
                        bi_det_pred_num,
                        bi_det_tp_num,
                        bi_det_fp_num,
                    ) = evaluate(self.cfgs, bi_det_preds, bi_det_ann)

                    # FIXME: remove .item() if multi-label case
                    det_gt_nums_tot += bi_det_gt_num
                    det_tp_nums_tot += bi_det_tp_num
                    # correct_nums_tot += correct_num
                    det_pred_nums_tot += bi_det_pred_num
                    det_fp_nums_tot += bi_det_fp_num

                    cls_pred_tot.append(bi_cls_pred)
                    cls_gt_tot.append(int(bi_det_ann[:, -1].max() > -1))

            # for Visualization - abnormal
            for viz_bi in range(len(data["fp"])):
                if data["bbox"][viz_bi, 0, -1] != -1:
                    break

            det_preds_viz = logits["preds"][viz_bi].detach().cpu().numpy()
            if len(det_preds_viz) != 0:  # No pred bbox
                # sigmoid
                det_preds_viz[:, -1] = 1 / (1 + np.exp(-1 * det_preds_viz[:, -1]))
            else:
                det_preds_viz = np.ones((1, 6)) * -1

            det_anns_viz = data["bbox"][viz_bi].detach().cpu().numpy()

            val_viz = {
                "fp": data["fp"][viz_bi],
                "img": data["img"][viz_bi].numpy(),
                "pred": det_preds_viz,
                "ann": det_anns_viz,
            }

        # NOTE: Reduce CPU-GPU Synchonization (.item() calls or printing CUDA tensors)
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

        val_record = {
            "loss": (losses_tot / (nums_tot + 1e-5)),
            "det_prec": det_pc.mean(),
            "det_recl": det_rc.mean(),
            "det_fppi": det_fppi.mean(),
            "det_f1": 2
            * det_pc.mean()
            * det_rc.mean()
            / (det_pc.mean() + det_rc.mean() + 1e-5),
            "cls_auc": cls_auc,
            "cls_sens": cls_sens,
            "cls_spec": cls_spec,
        }

        return val_record, val_viz


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def gather_helper(target):
    target_gather = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
    dist.all_gather(target_gather, target, async_op=False)
    target = torch.cat(target_gather, 0)

    return target
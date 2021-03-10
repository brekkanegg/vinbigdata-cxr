import torch
from .losses import *


def get_optimizer(cfgs, parameters):
    cfgs_opts = cfgs["model"]["opts"]

    if cfgs_opts["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            params=parameters,
            lr=cfgs_opts["learning_rate"],
            betas=(cfgs_opts["beta1"], cfgs_opts["beta2"]),
            weight_decay=cfgs_opts["weight_decay"],
        )
    elif cfgs_opts["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            params=parameters,
            lr=cfgs_opts["learning_rate"],
            betas=(cfgs_opts["beta1"], cfgs_opts["beta2"]),
            weight_decay=cfgs_opts["weight_decay"],
        )

    elif cfgs_opts["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params=parameters,
            lr=cfgs_opts["learning_rate"],
            momentum=cfgs_opts["beta1"],
            weight_decay=cfgs_opts["weight_decay"],
        )
    else:
        raise NotImplementedError("Invalid Optimizer", cfgs_opts["optimizer"])

    return optimizer


def get_scheduler(cfgs, optimizer):
    cfgs_opts = cfgs["model"]["opts"]
    if cfgs_opts["scheduler"] == "cosineWR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=10, T_mult=1
        )
    elif cfgs_opts["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=100
        )

    elif cfgs_opts["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    else:
        scheduler = None

    return scheduler


def calc_loss(cfgs, device, data, logits):
    cfgs_loss = cfgs["model"]["loss"]

    criterion = FocalLoss(cfgs)
    loss_det = criterion(logits, data["bbox"].to(device))

    if cfgs_loss["cls_weight"] > 0:
        aux_cls_criterion = nn.BCEWithLogitsLoss()
        data_cls = data["bbox"][:, :, -1][:, 0].unsqueeze(-1)
        data_cls = (data_cls > -1).float()
        loss_cls = aux_cls_criterion(logits["aux_cls"], data_cls.to(device))

        return loss_det, loss_cls

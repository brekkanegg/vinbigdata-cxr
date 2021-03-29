import torch


def get_optimizer(cfgs, parameters):
    cfgs_opts = cfgs["meta"]["opts"]

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
    cfgs_opts = cfgs["meta"]["opts"]
    if cfgs_opts["scheduler"] == "cosineWR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=5, T_mult=1
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

import matplotlib

matplotlib.use("Agg")  # tensorboardX
import os, sys
import torch
import random
import numpy as np
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch.backends.cudnn as cudnn


@hydra.main(config_path="conf", config_name="config")
def main(cfgs: DictConfig):

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfgs["gpu"]])
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # For Reproductibility
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(52)
    np.random.seed(52)
    torch.random.manual_seed(52)

    # Settings
    torch.cuda.set_device(0)

    if cfgs["run"] == "train":
        from scripts.train import Trainer

        Trainer(cfgs).do_train()

    elif cfgs["run"] == "val":
        from scripts.validate import Validator

        Validator(cfgs).do_validate()

    elif cfgs["run"] == "test":  # submission file 만들기
        from scripts.test import Testor

        Testor(cfgs).do_test()

    elif cfgs["run"] == "train_cls":
        from scripts.train_cls import Trainer

        Trainer(cfgs).do_train()

    elif cfgs["run"] == "val_cls":
        from scripts.validate_cls import Validator

        Validator(cfgs).do_validate()

    elif cfgs["run"] == "test_only_class":  # submission file 만들기
        from scripts.test_only_class import Testor

        Testor(cfgs).do_test()

    # TODO:
    # elif cfgs["run"] == "testcls":  # submission file 만들기
    #     from scripts.testcls import Testor

    #     Testor(cfgs).do_test()


if __name__ == "__main__":
    main()

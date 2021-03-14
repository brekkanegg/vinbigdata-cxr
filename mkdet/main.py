"""
example) python -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 2 main.py --run train --dryrun True --gpu 6 7


python -m torch.distributed.launch --help
usage: launch.py [-h] [--nnodes NNODES] [--node_rank NODE_RANK]
                 [--nproc_per_node NPROC_PER_NODE] [--master_addr MASTER_ADDR]
                 [--master_port MASTER_PORT] [--use_env] [-m] [--no_python]

Distributed Data Parallel
node: machine(server)
rank: gpu
world_size: tot gpu num

[Example] hydra 유의
- $python main.py gpu=[0], model=deeplab, model.inputs.image_size=1024 
- DDP + Hydra : 현재 같이 못 씀, 추후 수정

[Script 설명]
- train.py : 학습 코드
- validate.py: train.py 에서 validation 하는 부분만 따로 빼놓음
- test.py: csv 별로 validate 하고 metric 구하기
- inference.py: (추후 구현) label 없는 경우 inference

"""

import argparse
import matplotlib

matplotlib.use("Agg")  # tensorboardX
import os, sys
import json
import torch
import random
import numpy as np
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

from collections import OrderedDict

import torch.distributed as dist
import torch.backends.cudnn as cudnn


@hydra.main(config_path="conf", config_name="config")
def main(cfgs: DictConfig):

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfgs["gpu"]])
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # For Reproductibility
    cudnn.deterministic = cfgs["determ"]
    cudnn.benchmark = not cfgs["determ"]
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

    elif cfgs["run"] == "test":  # 'test' 와 동일,  /train/trainval/val(test)
        from scripts.test import Testor

        Testor(cfgs).do_test()


if __name__ == "__main__":
    main()

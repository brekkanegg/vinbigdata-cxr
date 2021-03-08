from .writer import Writer
from .logger import Logger


def get_logger(cfgs):
    log_dir = cfgs["save_dir"]
    run = cfgs["run"]
    out = Logger(log_dir=log_dir, run=run)
    return out


def get_writer(cfgs):
    out = Writer(cfgs)
    return out

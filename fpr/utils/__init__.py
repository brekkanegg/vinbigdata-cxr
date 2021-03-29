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


def convert_time(time):
    # time = float(input("Input time in seconds: "))
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time

    # ("%dD:%dH:%dM:%dS" % (day, hour, minutes, seconds))
    return "%1dD %2dH %2dM" % (day, hour, minutes)
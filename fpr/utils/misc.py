import yaml
import os
from collections import OrderedDict


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


def set_save_dir(cfgs):
    save_dict = OrderedDict()
    save_dict["fold"] = cfgs["fold"]
    if cfgs["memo"] is not None:
        save_dict["memo"] = cfgs["memo"]  # 1,2,3
    specific_dir = ["{}-{}".format(key, save_dict[key]) for key in save_dict.keys()]
    save_dir = os.path.join(cfgs["save_dir"], "_".join(specific_dir))

    return save_dir
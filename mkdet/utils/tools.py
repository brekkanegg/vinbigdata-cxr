import yaml
import os


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


# def save_cfg(cfg, cfg_dir):
#     os.makedirs(os.path.dirname(cfg_dir), exist_ok=True)
#     with open(cfg_dir, "w") as f:
#         _ = yaml.dump(cfg, f)
#     return


# def load_cfg(cfg_dir):
#     with open(cfg_dir, "rb") as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
#     return cfg


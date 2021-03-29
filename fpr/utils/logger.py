import os, sys
import time
import numpy as np
import torch

# from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, swidth=12, run="train"):
        self.log_dir = log_dir
        self.swidth = swidth
        self.run = run

        # self.terminal = sys.stdout  # stdout
        self.txt_file = open(self.log_dir + "/log.{}.txt".format(self.run), "a")

    #     self.file = None
    #     self.open()  # mode="a")

    # def open(self, mode=None):
    #     if mode is None:
    #         mode = "w"
    #     file = os.path.join(self.log_dir, "log.{}.txt".format(self.run))
    #     self.file = open(file, mode)

    def write(self, message, txt_write=True):
        # if "\r" in message:
        #     is_file = 0

        # if is_terminal == 1:
        sys.stdout.write(message)
        sys.stdout.flush()
        # time.sleep(1)

        # if is_file == 1:
        if txt_write:
            self.txt_file.write(message)
            self.txt_file.flush()

    def flush(self):
        pass

    def log_header(self, header_columns=None):

        log_messages = (" {: ^{w}} |" * len(header_columns)).format(
            *header_columns, w=self.swidth
        )
        self.write(log_messages)
        self.write("\n")
        self.write("-" * len(log_messages) + "\n")

    def log_result(self, result, txt_write=False):
        # print("\n", end="", flush=True)
        # self.write("\r")

        # FIXME: String, Integer, Time, Float
        # swidth = 16
        log_messages = "\r"
        for r in result:
            if (
                (type(r) == float)
                or (type(r) == np.float64)
                or (type(r) == torch.Tensor)
            ):
                log_messages += " {: ^{w}.4f} |".format(r, w=self.swidth)
            else:
                log_messages += " {: ^{w}} |".format(r, w=self.swidth)

        self.write(log_messages, txt_write)

    def log_train(self, train_result):

        raise NotImplementedError

    def log_val(self, val_result):
        raise NotImplementedError

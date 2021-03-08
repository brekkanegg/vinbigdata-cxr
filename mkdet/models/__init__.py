from .efficientdet.model import *


def get_model(cfgs):

    if cfgs["model"]["model"]["name"] == "EfficientDet":
        model = EfficientDet(cfgs)
    else:
        raise

    return model

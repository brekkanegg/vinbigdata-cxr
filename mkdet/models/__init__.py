def get_model(cfgs, pretrained=True):
    if cfgs["meta"]["model"]["name"] == "EfficientDet":
        from .efficientdet.model import EfficientDet

        model = EfficientDet(cfgs, pretrained=pretrained)

    else:
        raise

    return model

def get_model(cfgs, pretrained=True):
    from .efficientdet.model import EfficientDet

    model = EfficientDet(cfgs, pretrained=pretrained)

    return model

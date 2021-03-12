def get_model(cfgs):
    import os, sys

    sys.path.append(os.path.dirname(__file__))

    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
    from effdet.efficientdet import HeadNet

    config = get_efficientdet_config("tf_efficientdet_d5")
    config.num_classes = cfgs["model"]["inputs"]["num_classes"]
    ims = cfgs["model"]["inputs"]["image_size"]
    config.image_size = [ims, ims]
    config.norm_kwargs = dict(eps=0.001, momentum=0.01)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    model = DetBenchTrain(net, config)

    return model

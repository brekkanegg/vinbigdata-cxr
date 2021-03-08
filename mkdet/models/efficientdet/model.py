"""
based on https://github.com/tristandb/EfficientDet-PyTorch
and https://github.com/toandaominh1997/EfficientDet.Pytorch
"""
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import nms

from efficientnet_pytorch import EfficientNet

# from .efficientnet import EfficientNet
from .bifpn import BiFPN
from .retinahead import RetinaHead
from .module import (
    RegressionModel,
    ClassificationModel,
    Anchors,
    ClipBoxes,
    BBoxTransform,
    AuxClassificationModel,
)

# EfficientNet configuration of num_channels, num_layers(bifpn), num_layers(seg)
EfficientNet_CFG = {
    "b0": [64, 2, 3],  # 512
    "b1": [88, 3, 3],  # 640
    "b2": [112, 4, 3],  # 768
    "b3": [160, 5, 4],  # 896
    "b4": [224, 6, 4],  # 1024
    "b5": [288, 7, 4],  # 1280
    "b6": [384, 8, 5],  # 1408
}


class EfficientDet(nn.Module):
    def __init__(self, cfgs, feature_net="b4"):

        super(EfficientDet, self).__init__()

        self.cfgs = cfgs

        W_bifpn = EfficientNet_CFG[feature_net][0]
        D_bifpn = EfficientNet_CFG[feature_net][1]
        L_head = EfficientNet_CFG[feature_net][2]

        self.num_channels = W_bifpn
        self.f_start = cfgs["model"]["model"]["feat_start_layer"]

        efficientnet = EfficientNet.from_pretrained("efficientnet-" + feature_net)

        blocks = []
        fpn_channels = []
        for block in efficientnet._blocks:
            blocks.append(block)
            if block._depthwise_conv.stride == [2, 2]:
                fpn_channels.append(block._project_conv.out_channels)
                if len(fpn_channels) >= 4:
                    break

        self.efficientnet = nn.Sequential(
            efficientnet._conv_stem, efficientnet._bn0, *blocks
        )

        # [1:4],  # in_channels=fpn_channels[1:],
        self.bifpn = BiFPN(
            in_channels=fpn_channels[self.f_start : self.f_start + 3],
            num_channels=EfficientNet_CFG[feature_net][0],
            num_layers=EfficientNet_CFG[feature_net][1],
        )
        self.bbox_head = RetinaHead(
            cfgs=self.cfgs,
            in_channels=W_bifpn,
            stacked_convs=L_head,
        )

        pyramid_levels = [3, 4, 5, 6, 7]
        if self.f_start == 0:
            pyramid_levels = [2, 3, 4, 5, 6]

        self.anchors = Anchors(pyramid_levels=pyramid_levels)
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        # if self.cfgs["model"]["loss"]["cls_weight"] > 0:
        self.aux_classifier = AuxClassificationModel(self.num_channels)

        self._init_weights()
        # NOTE: RetinaHead initialize 따로
        self.bbox_head.init_weights()

    def forward(self, inputs, mode="train", *kwargs):

        outputs_dict = {}
        x = self.efficientnet[0](inputs)
        x = self.efficientnet[1](x)

        # Forward batch through backbone
        features = []
        for block in self.efficientnet[2:]:
            x = block(x)
            if block._depthwise_conv.stride == [2, 2]:
                features.append(x)

        x_feat = self.bifpn(features[self.f_start : self.f_start + 3])
        outs = self.bbox_head(x_feat)

        classification = torch.cat([out for out in outs[0]], dim=1)
        regression = torch.cat([out for out in outs[1]], dim=1)
        anchors = self.anchors(inputs)

        outputs_dict["classification"] = classification
        outputs_dict["regression"] = regression
        outputs_dict["anchors"] = anchors

        if mode != "train":
            det_th = self.cfgs["model"]["val"]["det_th"]
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > det_th)[:, :, 0]

            preds = {}
            for bi in range(scores.shape[0]):
                bi_scores_over_thresh = scores_over_thresh[bi, :]
                if bi_scores_over_thresh.sum() == 0:
                    # print("No boxes to NMS")
                    # no boxes to NMS, just return
                    preds[bi] = torch.zeros((0, 6))  # bbox(4), class(1), score(1)
                    continue

                bi_transformed_anchors = transformed_anchors[bi, :, :]
                bi_classification = classification[bi, bi_scores_over_thresh, :]
                bi_transformed_anchors = transformed_anchors[
                    bi, bi_scores_over_thresh, :
                ]
                bi_scores = scores[bi, bi_scores_over_thresh, :]
                # FIXME: memory constrarint -- cpu()
                bi_anchors_nms_idx = nms(
                    bi_transformed_anchors,
                    bi_scores[:, 0],
                    iou_threshold=self.cfgs["model"]["val"]["nms_th"],
                )
                bi_nms_scores, bi_nms_class = bi_classification[
                    bi_anchors_nms_idx, :
                ].max(dim=1)

                bi_pred_bbox = bi_transformed_anchors[bi_anchors_nms_idx, :]
                preds[bi] = torch.cat(
                    (
                        bi_pred_bbox,
                        bi_nms_class.unsqueeze(-1),
                        bi_nms_scores.unsqueeze(-1),
                    ),
                    dim=1,
                )

            outputs_dict["preds"] = preds

        if self.cfgs["model"]["loss"]["cls_weight"] > 0:
            aux_cls = self.aux_classifier(x_feat[-1])[:, :, 0, 0]
            outputs_dict["aux_cls"] = aux_cls

        return outputs_dict

        # # Original
        # scores_over_thresh = (scores > self.args.det_threshold)[0, :, 0]
        # if scores_over_thresh.sum() == 0:
        #     print("No boxes to NMS")
        #     # no boxes to NMS, just return
        #     return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
        # classification = classification[:, scores_over_thresh, :]
        # transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        # scores = scores[:, scores_over_thresh, :]

        # anchors_nms_idx = nms(
        #     transformed_anchors[0, :, :],
        #     scores[0, :, 0],
        #     iou_threshold=self.args.iou_threshold,
        # )
        # nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        # return [
        #     nms_scores,
        #     nms_class,
        #     transformed_anchors[0, anchors_nms_idx, :],
        # ]

    # def extract_feat(self, img):
    #     """
    #     Directly extract features from the backbone+neck
    #     """
    #     x = self.efficientnet(img)
    #     x = self.neck(x[-5:])
    #     return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

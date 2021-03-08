import torch.nn as nn
import torch.nn.functional as F


# from .module import ConvModule, xavier_init
import torch


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.


    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        freeze_bn=False,
    ):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        freeze_bn=False,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, num_channels=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_td = DepthwiseConvBlock(num_channels, num_channels)
        self.p4_td = DepthwiseConvBlock(num_channels, num_channels)
        self.p5_td = DepthwiseConvBlock(num_channels, num_channels)
        self.p6_td = DepthwiseConvBlock(num_channels, num_channels)

        self.p4_out = DepthwiseConvBlock(num_channels, num_channels)
        self.p5_out = DepthwiseConvBlock(num_channels, num_channels)
        self.p6_out = DepthwiseConvBlock(num_channels, num_channels)
        self.p7_out = DepthwiseConvBlock(num_channels, num_channels)

        # Initialize W as 1
        self.down_w = nn.Parameter(torch.ones(2, 4))
        self.up_w = nn.Parameter(torch.ones(3, 3))
        self.up7_w = nn.Parameter(torch.ones(2, 1))
        self.act = nn.ReLU()

    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs

        # Calculate Top-Down Pathway
        down_w = self.act(self.down_w)
        down_w /= torch.sum(down_w, dim=0) + self.epsilon

        p6_td = self.p6_td(
            down_w[0, 0] * p6_x
            + down_w[1, 0]
            * F.interpolate(p7_x, scale_factor=2, recompute_scale_factor=False)
        )
        p5_td = self.p5_td(
            down_w[0, 1] * p5_x
            + down_w[1, 1]
            * F.interpolate(p6_x, scale_factor=2, recompute_scale_factor=False)
        )
        p4_td = self.p4_td(
            down_w[0, 2] * p4_x
            + down_w[1, 2]
            * F.interpolate(p5_x, scale_factor=2, recompute_scale_factor=False)
        )
        p3_td = self.p3_td(
            down_w[0, 3] * p3_x
            + down_w[1, 3]
            * F.interpolate(p4_x, scale_factor=2, recompute_scale_factor=False)
        )

        # Calculate Bottom-Up Pathway
        up_w = self.act(self.up_w)
        up_w /= torch.sum(up_w, dim=0) + self.epsilon
        p3_out = p3_td
        p4_out = self.p4_out(
            up_w[0, 0] * p4_x
            + up_w[1, 0] * p4_td
            + up_w[2, 0]
            * F.interpolate(p3_out, scale_factor=0.5, recompute_scale_factor=False)
        )  # # Todo: check 0.5 works
        p5_out = self.p5_out(
            up_w[0, 1] * p5_x
            + up_w[1, 1] * p5_td
            + up_w[2, 1]
            * F.interpolate(p4_out, scale_factor=0.5, recompute_scale_factor=False)
        )
        p6_out = self.p6_out(
            up_w[0, 2] * p6_x
            + up_w[1, 2] * p6_td
            + up_w[2, 2]
            * F.interpolate(p5_out, scale_factor=0.5, recompute_scale_factor=False)
        )  #

        up7_w = self.act(self.up7_w)
        up7_w /= torch.sum(up7_w, dim=0) + self.epsilon
        p7_out = self.p7_out(
            up7_w[0, 0] * p7_x
            + up7_w[1, 0]
            * F.interpolate(p6_out, scale_factor=0.5, recompute_scale_factor=False)
        )  # Fixme

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPN(nn.Module):
    def __init__(self, in_channels, num_channels=64, num_layers=2, epsilon=0.0001):
        super(BiFPN, self).__init__()
        self.p3 = nn.Conv2d(
            in_channels[0], num_channels, kernel_size=1, stride=1, padding=0
        )
        self.p4 = nn.Conv2d(
            in_channels[1], num_channels, kernel_size=1, stride=1, padding=0
        )
        self.p5 = nn.Conv2d(
            in_channels[2], num_channels, kernel_size=1, stride=1, padding=0
        )

        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(
            in_channels[2], num_channels, kernel_size=3, stride=2, padding=1
        )

        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = ConvBlock(
            num_channels, num_channels, kernel_size=3, stride=2, padding=1
        )

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(num_channels))
        self.bifpn = nn.Sequential(*bifpns)

        # self._init_weights()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c5)
        p7_x = self.p7(p6_x)

        features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        return self.bifpn(features)
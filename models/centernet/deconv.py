import math

import torch
import torch.nn as nn
from detectron2.layers.deform_conv import DeformConv, ModulatedDeformConv


class DeformConvWithOff(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
    ):
        super(DeformConvWithOff, self).__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcn = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        offset = self.offset_conv(input)
        output = self.dcn(input, offset)
        return output


class ModulatedDeformConvWithOff(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
    ):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = ModulatedDeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output


class DeconvLayer(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        deconv_kernel,
        deconv_stride=2,
        deconv_pad=1,
        deconv_out_pad=0,
        modulate_deform=True,
    ):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConvWithOff(
                in_planes, out_planes, kernel_size=3, deformable_groups=1
            )
        else:
            self.dcn = DeformConvWithOff(in_planes, out_planes, kernel_size=3, deformable_groups=1)

        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class CenternetDeconv(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg):
        super(CenternetDeconv, self).__init__()
        # modify into config
        channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL
        deconv_kernel = cfg.MODEL.CENTERNET.DECONV_KERNEL
        modulate_deform = cfg.MODEL.CENTERNET.MODULATE_DEFORM
        self.deconv1 = DeconvLayer(
            channels[0],
            channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[1],
            channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[2],
            channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upsample_func=None,
                 scale=4):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True))

        # output conv.
        self.conv_out = nn.Conv2d(4 * ((4//scale)**2), out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        print(hr_prev_tran.shape, lr_curr.shape)
        # exit(0)
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        out = self.resblocks(out)
        out = self.conv_up_cheap(out)
        print(out.shape)
        # exit(0)
        out = self.conv_out(out)
        # out += self.upsample_func(lr_curr)

        return out

def test_SRNet():
    scale = 4
    net = SRNet(scale=scale)
    lr_input = torch.rand((1, 3, 32, 32))
    hr_input = torch.rand((1, 3*scale*scale, 32, 32))
    output = net(lr_input, hr_input)
    print(output.shape)
    pass


def test():
    a = torch.rand((1, 7, 32, 32))
    b = a[:, :-1, ...]
    c = a[:, 1:, ...]
    print(b.shape, c.shape, a.shape)
    # print(b.reshape(6))
    # print(c.reshape(6))
    # print(a.reshape(7))
    P = nn.MaxPool2d(2, 2)
    d = P(a)
    print(d.shape)
    pass

def test2():
    a = torch.rand((1, 19, 1, 1))
    fw = a[:, :10-1, ...]
    bw = a[:, 10:, ...].flip(1)
    print(a.reshape(19))
    print(fw.reshape(9))
    print(bw.reshape(9))

def test_inter():
    size = 4
    a = torch.rand((2, 1, 3, size, size))
    a = (a+1.0)/2
    b = torch.zeros(2, 1, 3, size//2, size//2)
    print(b.shape)
    for i in range(2):
        lr = F.interpolate(a[i, ...], scale_factor=1/2, mode="bicubic", align_corners=False)
        b[i, ...] = lr
    # lr_data = F.interpolate(a, scale_factor=1/2, mode="bicubic", align_corners=False)
    print(b.shape)
    print(a)
    print(b)


def test_tensor():
    img = Image.open('d:/workroom/testroom/old.png')
    # img.show()
    print(img.size)
    img_tensor = transforms.ToTensor()(img)
    print(img_tensor.shape)

if __name__ == "__main__":
    # test_SRNet()
    # test()
    # test2()
    test_inter()
    # test_tensor()

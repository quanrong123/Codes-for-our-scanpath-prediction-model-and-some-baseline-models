import torch
import torch.nn as nn
from parameters import *
import os
import numpy as np

def conv2d(in_channels, out_channels, kernel_size = 3, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

def deconv2d(in_channels, out_channels, kernel_size = 3, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

def relu(inplace = True):
    return nn.ReLU(inplace)

def maxpool2d():
    return nn.MaxPool2d(2)

def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_deconv_layers(cfg):
    layers = []
    in_channels = 512
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D':[512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}

def encoder():
    return make_conv_layers(cfg['E'])

def decoder():
    return make_deconv_layers(cfg['D'])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.mymodules = nn.ModuleList([
            deconv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])

    def upsampling(self, x, image_h, image_w):
        m = nn.UpsamplingBilinear2d(size=[image_h, image_w])
        x = m(x)
        return x

    def forward(self, x, image_h, image_w):
        x = self.encoder(x)

        x = self.decoder[0](x)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        f6 = self.decoder[4](x)
        x = self.decoder[5](f6)
        x = self.decoder[6](x)
        x = self.decoder[7](x)
        x = self.decoder[8](x)
        x = self.decoder[9](x)
        x = self.decoder[10](x)
        f7 = self.decoder[11](x)
        x = self.decoder[12](f7)
        x = self.decoder[13](x)
        x = self.decoder[14](x)
        x = self.decoder[15](x)
        x = self.decoder[16](x)
        x = self.decoder[17](x)
        f8 = self.decoder[18](x)
        x = self.decoder[19](f8)
        x = self.decoder[20](x)
        x = self.decoder[21](x)
        x = self.decoder[22](x)
        f9 = self.decoder[23](x)
        x = self.decoder[24](f9)
        x = self.decoder[25](x)
        x = self.decoder[26](x)
        x = self.decoder[27](x)
        f10 = self.decoder[28](x)
        x = self.decoder[29](f10)
        del x
        f6 = torch.squeeze(self.upsampling(f6, image_h, image_w).data)
        f7 = torch.squeeze(self.upsampling(f7, image_h, image_w).data)
        f8 = torch.squeeze(self.upsampling(f8, image_h, image_w).data)
        f9 = torch.squeeze(self.upsampling(f9, image_h, image_w).data)
        f10 = torch.squeeze(self.upsampling(f10, image_h, image_w).data)
        return torch.cat((f6, f7, f8, f9, f10), 0)  #


net_params_path = './gen_modelWeights0090'
net_params_pathDir = os.listdir(net_params_path)
net_params_pathDir.sort()

def gen_SalGAN():
    net = Generator()
    params = net.state_dict()
    n1 = 0
    pretrained_dict = {}
    for k, v in params.items():
        single_file_name = net_params_pathDir[n1]
        single_file_path = os.path.join(net_params_path, single_file_name)
        pa = np.load(single_file_path)
        pa = torch.from_numpy(pa)
        pretrained_dict[k] = pa
        n1 += 1
    params.update(pretrained_dict)
    net.load_state_dict(params)
    return net







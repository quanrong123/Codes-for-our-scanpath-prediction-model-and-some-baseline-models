import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from parameters import *
import torch
torch.cuda.set_device(gpu)
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import scipy.io as scio

from generator import gen_SalGAN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
m = nn.BatchNorm2d(1475, affine=True).cuda()

imgspathdir = os.listdir(imgspath)
imgspathdir.sort()
num_images = len(imgspathdir)
spspathdir = os.listdir(spspath)
spspathdir.sort()


net = gen_SalGAN()
net = net.cuda()

def get_whole_image_features(image, net, image_h, image_w):
    image1 = torch.Tensor(transform(image)).cuda()
    image = transform(image).float()
    image = Variable(torch.unsqueeze(image, 0)).cuda()
    features = net(image, image_h, image_w)
    features = Variable(torch.unsqueeze(torch.cat((image1, features), 0), 0))
    features = torch.squeeze(m(features)).data
    return features

for i in range(num_images):
    image_name = imgspathdir[i]
    image_path = os.path.join(imgspath, image_name)
    image = Image.open(image_path)
    image_h = image.size[1]
    image_w = image.size[0]

    features = get_whole_image_features(image, net, image_h, image_w)

    sps_name = spspathdir[i]
    sps_path = os.path.join(spspath, sps_name)
    sps = scio.loadmat(sps_path)
    resname = os.path.join(respath, sps_name)

    sp_seg = sps['sp_seg']
    spstats = sps['spstats']
    num_sps = np.max(sp_seg)
    sp_features = torch.Tensor(num_sps, 1475)
    for j in range(num_sps):
        a,b = np.where(sp_seg==(j+1))
        c = features[:,a,b]
        d = torch.mean(c, dim=1)
        sp_features[j] = d
    sp_features = sp_features.numpy()
    scio.savemat(resname, {"sp_features":sp_features})
    del features, sp_features
    torch.cuda.empty_cache()
    #print(1)






















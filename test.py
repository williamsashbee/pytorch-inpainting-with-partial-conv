import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/home/washbee1/celeba-kaggle')
parser.add_argument('--maskroot', type=str, default='mask')
parser.add_argument('--snapshot', type=str, default='../inpainting-inuse/saves-data-aug-lm/ckpt/5690000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(args.root, args.maskroot, img_transform, mask_transform, 'test')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, dataset_val, device, 'result.jpg')

import argparse
import torch
from PIL import Image
from torchvision import transforms

import opt
from places2 import Places2
from net import PConvUNet
from util.io import load_ckpt

import torchvision
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

def demo(dataset, device):

    for path in dataset.paths:
        ind = dataset.getIndex(path)
        image = dataset[ind]
        crop_img = image[:,0:512, 0:450]
        torchvision.utils.save_image(crop_img, path)



parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/home/washbee1/data1024x1024-512-temp/data_large/train')
parser.add_argument('--image_size', type=int, default=512)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)

size = (args.image_size, args.image_size)


img_tf = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


dataset_val = Places2(args.root, None, img_tf, None, 'demo')

demo( dataset_val,  'demo.jpg')





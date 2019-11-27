import argparse
import torch
from torchvision import transforms
from torchvision.transforms.functional import rotate
import opt
from torchvision.utils import save_image

from PIL import Image
from glob import glob

from util.image import unnormalize

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

class PreprocessDemo(torch.utils.data.Dataset):

    def __init__(
            self,
            img_root,
            img_transform
    ):
        super(PreprocessDemo, self).__init__()
        self.img_transform = img_transform
        self.paths = glob('{:s}/*.jpg'.format(img_root))

    def transformdata(self):
        for path in self.paths:
            gt_img = Image.open(path)
            gt_img = self.img_transform(gt_img.convert('RGB'))
            gt_img = tuple([gt_img])
            gt_img = torch.stack(gt_img)
            #img = unnormalize(gt_img)
            save_image(gt_img, path)

    def rotate(self,rotation):
        for path in self.paths:
            gt_img = Image.open(path)
            gt_img = rotate(gt_img,rotation,resample=False,expand=False,center=None)
            gt_img = self.img_transform(gt_img.convert('RGB'))
            gt_img = tuple([gt_img])
            gt_img = torch.stack(gt_img)
            #img = unnormalize(gt_img)
            save_image(gt_img, path)

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/home/washbee1/phone-images')
parser.add_argument('--mask_root', type=str, default='/home/washbee1/masks')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = ( 218, 178)
#img_transform = transforms.Compose(
#    [transforms.Resize(size=size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

img_tf = transforms.Compose(
    [
        transforms.Resize(size = (2180,2180)),
        transforms.RandomResizedCrop(2180, scale=(0.8, .9), ratio=(.9, 1.1), interpolation=2),
        transforms.RandomRotation(0, resample=False, expand=False),
        transforms.Resize(size=size),
        transforms.ToTensor()
        #transforms.Normalize(mean=opt.MEAN, std=opt.STD)
    ]
)


dataset_val = PreprocessDemo(args.root,  img_tf)


dataset_val.transformdata()
#dataset_val.rotate(15)

#dataset_val = PreprocessDemo(args.mask_root,  img_tf)


#dataset_val.transformdata()
#dataset_val.rotate(30)

#torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None, fill=0)



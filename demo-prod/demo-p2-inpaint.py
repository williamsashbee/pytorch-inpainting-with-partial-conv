import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from net import PConvUNet
from util.io import load_ckpt

from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize
from PIL import Image
import os


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"

def demo(model, dataset, device, filename):

    for path in dataset.paths:
        gt_img = Image.open(path)
        gt_img = dataset.img_transform(gt_img.convert('RGB'))
        for maskpath in dataset.mask_paths:
            base = os.path.basename(maskpath)
            base = os.path.splitext(base)[0]
            history = gt_img
            if base[:-2] in path:

                mask = Image.open(maskpath)
                mask = dataset.mask_transform(mask.convert('RGB'))
                mask = (mask > .1).type(torch.FloatTensor)

                #                (gt_img,mask) = zip(*[gt_img,mask])
                gt_img = torch.reshape(gt_img, (1, 3, 256, 256))

                mask = torch.reshape(mask, (1, 3, 256, 256))

#                gt = torch.stack(gt)

                output  = None
                with torch.no_grad():
                    output, _ = model(gt_img.to(device), mask.to(device))

                output = output.to(torch.device('cpu'))

                grid = make_grid(
                    torch.cat((unnormalize(gt_img), mask, unnormalize(gt_img*mask) ,unnormalize(output)
                    ), dim=0))
                save_image(grid, base+'_out.jpg')



parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='../demo-prod')
parser.add_argument('--maskroot', type=str, default='../demo-prod/demo-masks')
parser.add_argument('--snapshot', type=str, default='/home/washbee1/PycharmProjects/image_inpainting/targeted-training/saves-targeted-1/ckpt/6935000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)

img_tf = transforms.Compose(
    [
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.MEAN, std=opt.STD)
    ]
)

mask_tf = transforms.Compose(
    [transforms.Resize(size=size),
     transforms.ToTensor()])


dataset_demo = Places2(args.root, args.maskroot, img_tf, mask_tf, 'demo')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
demo(model, dataset_demo, device, 'demo.jpg')





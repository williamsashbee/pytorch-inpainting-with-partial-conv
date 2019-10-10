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



import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def demo(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(1)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)



parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/home/washbee1/phone-images')
parser.add_argument('--maskroot', type=str, default='/home/washbee1/masks')
parser.add_argument('--snapshot', type=str, default='../inpainting-inuse/1010000.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(args.root, args.maskroot, img_transform, mask_transform, 'demo')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
demo(model, dataset_val, device, 'demo.jpg')





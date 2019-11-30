import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.jpg'.format(img_root),
                              recursive=True)
        elif split == 'demo':
            self.paths = glob('{:s}/*.jpg'.format(img_root))
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))

        if mask_root is not None:
            self.mask_paths = glob('{:s}/*.jpg'.format(mask_root))

            self.N_mask = len(self.mask_paths)
        else:
            self.mask_paths = None
            self.N_mask = None

    def getIndex(self,path):
        ind = -1
        for p in self.paths:
            ind+=1
            if p == path:
                return ind
        return -1


    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
            mask = self.mask_transform(mask.convert('RGB'))
            return gt_img * mask, mask, gt_img
        return gt_img

    def __len__(self):
        return len(self.paths)

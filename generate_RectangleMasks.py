import argparse
import numpy as np
import random
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_rectangle(canvas):
    img_size = canvas.shape[-1]
    xi = np.random.randint(0,img_size-50)
    yi = np.random.randint(0,img_size-50)

    xlen = np.random.randint(40,img_size-xi-1)
    ylen = np.random.randint(40,img_size-yi-1)

    if xlen*ylen> img_size**2/5.0:
        xlen  = int(xlen/np.random.uniform(2,5))
        ylen  = int(ylen/np.random.uniform(2,5))


    canvas[xi:(xi+xlen), yi:(yi+ylen)] = 0
    return canvas


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='mask')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.N):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        mask = random_rectangle(canvas)
        print("save:", i, np.sum(mask))

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))

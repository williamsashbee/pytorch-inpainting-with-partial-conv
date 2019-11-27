import argparse
import numpy as np
import random
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_rectangle(canvas):
    img_size = canvas.shape[-1]
    xi = np.random.randint(0,img_size-(img_size/5))
    yi = np.random.randint(0,img_size- (img_size/5))

    xlen = np.random.randint(img_size/7,img_size)
    ylen = np.random.randint(img_size/7,img_size)

    while True:
        if  xi+xlen > img_size:
            xlen = int(xlen / 1.5)
        elif yi + ylen > img_size:
            ylen = int(ylen / 1.5)
        else:
            break

    while True:
        if xlen * ylen > img_size**2 / 3:
            if np.random.random_sample() > .5:
                xlen = int(xlen / 1.5)
            else:
                ylen = int(ylen / 1.5)
        else:
            break




    canvas[xi:(xi+xlen), yi:(yi+ylen)] = 0
    return canvas


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='mask-rect-large-temp')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.N):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        mask = random_rectangle(canvas)
        print("save:", i, np.sum(mask))

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))

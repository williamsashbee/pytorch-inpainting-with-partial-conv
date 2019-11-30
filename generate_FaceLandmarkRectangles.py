import argparse
import numpy as np
import random
from PIL import Image
from places2 import Places2
from imutils import face_utils
import argparse
import imutils
import time


import dlib
import cv2


action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_face_rectangle(args, path, canvas,i):
    img_size = canvas.shape[-1]
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    image = cv2.imread(path)
    image = imutils.resize(image, width=512,height=512)
    img = Image.fromarray(image * 255).convert('1')

    img.save('{:s}/{:06d}face.jpg'.format(args.save_dir, i))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),
                      (0, 255, 0), 1)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        xyList=[]
        for (_, (x, y)) in enumerate(shape):
            xyList.append((x,y))

        print(xyList)
        xyList = sorted(xyList, key=lambda point: point[0])
        print(xyList)
        sf = .5
        if len(xyList) == 5:
            i += 1
            canvas = np.ones((args.image_size, args.image_size)).astype("i")
            ymean = .5*(xyList[0][1]+xyList[1][1])
            xdiff = np.abs(xyList[0][0]-xyList[1][0])
            canvas[int(xyList[0][0]-sf*xdiff):int(xyList[1][0]+ sf*xdiff), int(ymean - sf*xdiff):int(ymean+sf*xdiff)] = 0
            canvas = canvas.T
            img = Image.fromarray(canvas * 255).convert('1')
            img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))

            i += 1
            canvas = np.ones((args.image_size, args.image_size)).astype("i")
            ymean = .5 * (xyList[3][1] + xyList[4][1])
            xdiff = np.abs(xyList[3][0] - xyList[4][0])
            canvas[int(xyList[3][0]-sf*xdiff):int(xyList[4][0]+sf*xdiff), int(ymean - sf*xdiff ):int(ymean + sf*xdiff )] = 0
            canvas = canvas.T
            img = Image.fromarray(canvas * 255).convert('1')
            img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))

            i += 1
            canvas = np.ones((args.image_size, args.image_size)).astype("i")
            ymin = np.max(np.array([xyList[0][1],xyList[1][1],xyList[3][1],xyList[4][1],]))
            xdiff = np.abs(xyList[3][0] - xyList[4][0])
            ymin -= int(sf*xdiff)
            ymax = int(xyList[2][1] + sf * xdiff)
            xmin = int(xyList[0][0] - sf * xdiff)
            xmax = int(xyList[4][0] + sf * xdiff)
            canvas[int(xmin):int(xmax), int(ymin):int(ymax)] = 0
            canvas = canvas.T
            img = Image.fromarray(canvas * 255).convert('1')
            img.save('{:s}/{:06d}.jpg'.format(args.save_dir, i))

    return canvas,i


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--N', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='mask-rect-large-hq-512')
    parser.add_argument('--root', type=str, default='../../../celeba-hq-512/data1024x1024-512/data_large/train')
    parser.add_argument("-p", "--shape-predictor",
                    help="path to facial landmark predictor", default = "shape_predictor_5_face_landmarks.dat")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset_train = Places2(args.root, None, None, None, 'demo')
    i = 0
    for path in dataset_train.paths:
        print(path)
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        mask,i = random_face_rectangle(args,path, canvas,i)
        #i=cnt
        print("save:", i, np.sum(mask))


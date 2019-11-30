from places2 import Places2
from imutils import face_utils
import argparse
import numpy as np
import dlib
import cv2

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def findMeans(args, paths):
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)
    endPath = np.min([len(paths), 1000])
    xVals = np.zeros((endPath, 5))
    yVals = np.zeros((endPath, 5))

    count = -1
    for path in paths:
        count+=1
        if count == endPath:
            break
        image = cv2.imread(path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            if len(rects) != 1:
                break


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

            xyList = sorted(xyList, key=lambda point: point[0])

            sf = .5
            if len(xyList) == 5:
                ind = 0
                for x, y in xyList:
                    xVals[count,ind] = x
                    yVals[count,ind] = y
                    ind+=1
    print(np.median(xVals,axis=0))
    print(np.median(yVals,axis=0))


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='mask-rect-large-temp--temp')
    parser.add_argument('--root', type=str, default='/home/washbee1/celeba-512-mix/data1024x1024-512/data_large/train')
    parser.add_argument("-p", "--shape-predictor",
                    help="path to facial landmark predictor", default = "/home/washbee1/PycharmProjects/image_inpainting/inpainting-inuse/shape_predictor_5_face_landmarks.dat")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset_train = Places2(args.root, None, None, None, 'demo')
    print(len(dataset_train.paths))

    findMeans(args, dataset_train.paths)


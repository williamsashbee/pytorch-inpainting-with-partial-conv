from places2 import Places2
from imutils import face_utils
import argparse

import dlib
import cv2


action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def crop_face(args, path, detector):
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        if len(rects) != 1:
            print(path, " not recognized")
            break
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        image = image[bY:bY + bH, bX:bX + bW, :]
        cv2.imwrite(path, image)




if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='mask-rect-large-hq-512')
    parser.add_argument('--root', type=str, default='/home/washbee1/celeba-hq-crop/data1024x1024/data_large/train')
    parser.add_argument("-p", "--shape-predictor",
                    help="path to facial landmark predictor", default = "../shape_predictor_5_face_landmarks.dat")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset_train = Places2(args.root, None, None, None, 'demo')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    for path in dataset_train.paths:
        crop_face(args, path, detector)


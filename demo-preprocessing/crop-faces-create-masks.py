from places2 import Places2
from imutils import face_utils
import argparse
import numpy as np
import dlib
import cv2
from PIL import Image


action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]

def generateMask(detector, predictor,image, args,filename,canvas):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]

    #print(base)
    # loop over the face detections
    for rect in rects:
        if len(rects) != 1:
            print(path, " not recognized")
            break

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        img = Image.fromarray(gray * 255).convert('1')
        img.save('{:s}.jpg'.format(args.save_dir+'/'+filename + '_face'))

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        xyList = []
        for (_, (x, y)) in enumerate(shape):
            xyList.append((x, y))

        # print(xyList)
        xyList = sorted(xyList, key=lambda point: point[0])
        # print(xyList)

        sf = .5
        i = 0
        if len(xyList) == 5:
            #canvas = np.ones((args.image_size, args.image_size)).astype("i")
            ymean = .5 * (xyList[0][1] + xyList[1][1])
            xdiff = np.abs(xyList[0][0] - xyList[1][0])
            canvas[int(xyList[0][0] - sf * xdiff):int(xyList[1][0] + sf * xdiff),
            int(ymean - sf * xdiff):int(ymean + sf * xdiff)] = 0
            canvas = canvas.T
            img = Image.fromarray(canvas * 255).convert('1')
            img.save('{:s}.jpg'.format(args.save_dir+'/'+filename+'_'+str(i)))

            i += 1
            canvas = np.ones(canvas.shape).astype("i")
            ymean = .5 * (xyList[3][1] + xyList[4][1])
            xdiff = np.abs(xyList[3][0] - xyList[4][0])
            canvas[int(xyList[3][0] - sf * xdiff):int(xyList[4][0] + sf * xdiff),
            int(ymean - sf * xdiff):int(ymean + sf * xdiff)] = 0
            canvas = canvas.T
            img = Image.fromarray(canvas * 255).convert('1')
            img.save('{:s}.jpg'.format(args.save_dir+'/'+filename + '_' + str(i)))

            i += 1
            canvas = np.ones(canvas.shape).astype("i")
            ymin = np.min(np.array([xyList[0][1], xyList[1][1], xyList[3][1], xyList[4][1]]))
            xdiff = np.abs(xyList[3][0] - xyList[4][0])
            ymin -= int(sf * xdiff)
            ymax = int(xyList[2][1] + 3.0 * sf * xdiff)
            xmin = int(xyList[0][0] - sf * xdiff)
            xmax = int(xyList[4][0] + sf * xdiff)
            canvas[int(xmin):int(xmax), int(ymin):int(ymax)] = 0
            canvas = canvas.T
            img = Image.fromarray(canvas * 255).convert('1')
            img.save('{:s}.jpg'.format(args.save_dir+'/'+filename + '_' + str(i)))


def crop_face(args, path, detector, predictor):
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]

    #print(base)
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
        canvas = np.ones((bH,bW)).astype('i')
        generateMask(detector, predictor, image, args, base, canvas)


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='../masks-Facial-targeted-rect')
    #parser.add_argument('--mask_root', type=str, default='./masks-Facial-targeted-rect')
    #parser.add_argument('--root', type=str, default='/home/washbee1/celeba-hq-crop/data1024x1024/data_large/train')
    parser.add_argument('--root', type=str, default='/home/washbee1/celeba-hq-crop-2/data1024x1024/data_large/train')

    parser.add_argument("-p", "--shape-predictor",
                    help="path to facial landmark predictor", default = "../shape_predictor_5_face_landmarks.dat")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset_train = Places2(args.root, None, None, None, 'demo')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    for path in dataset_train.paths:
        crop_face(args, path, detector, predictor)


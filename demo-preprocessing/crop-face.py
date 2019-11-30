from imutils import face_utils
import argparse
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",
                help="path to facial landmark predictor", default="../shape_predictor_5_face_landmarks.dat")
ap.add_argument("-i", "--image",
                help="path to input image", default="../with-jacket.jpg")

args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
rects = detector(gray, 0)

# check to see if a face was detected, and if so, draw the total
# number of faces on the frame
# loop over the face detections
for rect in rects:
    if len(rects) != 1:
        break

    # compute the bounding box of the face and draw it on the
    # frame
    bX, bY, bW, bH = face_utils.rect_to_bb(rect)
    print(image.shape)
    image = image[ bY:bY + bH , bX:bX + bW, :]
    print(image.shape)

    #
    #	cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),
    #		(0, 255, 0), 1)


    cv2.imwrite('../with-jacket-crop.jpg',image)


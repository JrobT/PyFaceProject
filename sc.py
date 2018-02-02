#!/usr/bin/env python3

"""
Screen-capture script for emotion recognition.

Use Webcam to capture an image and then use the classifier class to predict the
emotion that is present in the image.
"""

import numpy as np
import cv2
import dlib
import math
import os.path

from constants import PRED, HAAR, HAAR2, HAAR3, HAAR4, EMOTIONS_5
from emotion_recognition import SVM
from face_aligner import FaceAligner


print(__doc__)

# Set Face Detectors.
faceDet = cv2.CascadeClassifier(HAAR)
faceDet2 = cv2.CascadeClassifier(HAAR2)
faceDet3 = cv2.CascadeClassifier(HAAR3)
faceDet4 = cv2.CascadeClassifier(HAAR4)
faceDet5 = dlib.get_frontal_face_detector()  # dlib's face detector

# Build the required objects.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
predictor = dlib.shape_predictor(PRED)  # file must be in dir
fa = FaceAligner(predictor, desiredFaceWidth=380)
data = {}

font = cv2.FONT_HERSHEY_SIMPLEX
emojis = []
for index, emotion in enumerate(EMOTIONS_5):
    emojis.append(cv2.imread("emojis//{}.png".format(emotion)))


def get_face_recs(image):
    """."""
    detections = faceDet5(image, 1)

    haar_detections = []
    if not len(detections) > 0:  # dlib's detector will work over 50% of the time
        haar_detections = faceDet.detectMultiScale(image, scaleFactor=1.1,
                                                   minNeighbors=10, minSize=(5, 5),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
        haar_detections2 = faceDet2.detectMultiScale(image, scaleFactor=1.1,
                                                     minNeighbors=10, minSize=(5, 5),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        haar_detections3 = faceDet3.detectMultiScale(image, scaleFactor=1.1,
                                                     minNeighbors=10, minSize=(5, 5),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        haar_detections4 = faceDet4.detectMultiScale(image, scaleFactor=1.1,
                                                     minNeighbors=10, minSize=(5, 5),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

        if len(haar_detections) > 0:
            for (x, y, w, h) in haar_detections:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                detections.append(dlib_rect)
                break  # if found, no point in making another feature vector
        elif len(haar_detections2) > 0:
            for (x, y, w, h) in haar_detections2:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                detections.append(dlib_rect)
                break
        elif len(haar_detections3) > 0:
            for (x, y, w, h) in haar_detections3:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                detections.append(dlib_rect)
                break
        elif len(haar_detections4) > 0:
            for (x, y, w, h) in haar_detections4:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                detections.append(dlib_rect)
                break

    return detections


def get_landmarks(image):
    """As in SVMs.py, used to create feature vectors to train on."""
    detections = get_face_recs(image)

    # We may detect 0, 1 or many faces. Loop through each face detected.
    for i, j in enumerate(detections):
        # Draw facial landmarks with the predictor class.
        shape = predictor(image, j)
        xlist = []
        ylist = []

        # Store X and Y coordinates in separate lists.
        for i in range(1, 68):  # 68 because we're looking for 68 landmarks
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        # Find both coordinates for the centre of gravity (middle point).
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)

        # Calculate the distance from centre to other points in both axes.
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        # Condition the vectors.
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(math.pi*2))
    if len(detections) < 1:
        return "error"
    return landmarks_vectorised


# Build and train the classifier we're using.
SVM = SVM()
if (os.path.isfile('svm.pkl')):
    SVM.load()
else:
    SVM.train()
    SVM.save()

# Open video capture.
cap = cv2.VideoCapture(0)
if cap.isOpened() is False:
    print("[Err] Capture failed to open.")

cv2.namedWindow("test")
while (cap.isOpened()):
    # Capture frame-by-frame.
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow("test", frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Quit")
        break
    elif k % 256 == 32:
        # SPACE pressed
        lm = "error"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rect = []
        detections = get_face_recs(frame)
        for detection in detections:
            clahe_image = clahe.apply(gray)
            aligned = fa.align(clahe_image, detection)
            lm = get_landmarks(aligned)
            face_rect = detection
            break

        if lm is not "error":
            sample = np.array([lm])
            sample.reshape(1,  -1)
            emotion = SVM.predict(sample)
            print("Emotion detected: {}".format(emotion.capitalize()))
            cv2.putText(frame, emotion.capitalize(),
                        (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
            cv2.imshow("Frame", frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

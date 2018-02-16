#!/usr/bin/env python3
"""Python3 script to extract faces from an image.

After my database is complete, I must look to see if a face can be found within
the image. I do this by using OpenCV's HAAR Cascades to detect faces, since
object detection using Haar feature-based cascade classifiers is an effective
object detection method proposed by Paul Viola and Michael Jones in their
paper, ``Rapid Object Detection using a Boosted Cascade of Simple Features''
published in 2001. After I've ran the image path through the dlib classifier
and a face has still not been detected in the image, I use HAAR cascade's face
detectors to ensure I get I full range of images in my dataset.

This script should take a while, as it is processing around 6000 images and
running potentially 5 face detectors on the image. OpenCV resizing errors may
be expected for a few images, any other errors should be treated as unexpected.

Testing the effect of resizing images on classification accuracy. CNN requires
resizing.
"""

# Import packages.
import cv2  # OpenCV
import glob
import os
import time
import dlib


# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\nBeginning to extract faces from the images...".format(script_name))
start = time.clock()  # Start of the speed test. ``clock()'' is most accurate.

emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]  # The emotion list

# HAAR Cascade Face Classifiers.
haar = 'OpenCV_HAAR_CASCADES//haarcascade_frontalface_default.xml'
haar2 = 'OpenCV_HAAR_CASCADES//haarcascade_frontalface_alt2.xml'
haar3 = 'OpenCV_HAAR_CASCADES//haarcascade_frontalface_alt.xml'
haar4 = 'OpenCV_HAAR_CASCADES//haarcascade_frontalface_alt_tree.xml'

# Set Face Detectors.
faceDet = cv2.CascadeClassifier(haar)
faceDet2 = cv2.CascadeClassifier(haar2)
faceDet3 = cv2.CascadeClassifier(haar3)
faceDet4 = cv2.CascadeClassifier(haar4)
faceDet5 = dlib.get_frontal_face_detector()  # dlib's face detector


def resize(gray):
    """Return the given image, resized to 380x380."""
    dim = (380, 380)
    out = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
    # cv2.imwrite('resized.png', out)
    return out


def detect_faces(emotion):
    """Use the classifers to detect faces."""
    files = glob.glob('combined_dataset//%s//*' % emotion)

    filenumber = 0  # Keep track of the number of images in the final dataset
    for f in files:
        haar_detections = []
        facefeatures = []
        print("***> Detecting faces in file %s." % (f.replace("//", "/")))

        frame = cv2.imread(f)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale
        # cv2.imwrite('grayscale.png', gray)

        # Use detectors to detect faces in the frame.
        detections = faceDet5(gray, 1)

        if len(detections) > 0:  # dlib found faces
            for face in detections:
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()
                facefeatures.append([x, y, w, h])
        else:
            haar_detections = faceDet.detectMultiScale(gray, scaleFactor=1.1,
                                                       minNeighbors=10,
                                                       minSize=(5, 5),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections) > 0:  # HAAR Cascade found faces
                facefeatures = haar_detections

            if len(facefeatures) == 0:
                haar_detections2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1,
                                                             minNeighbors=10,
                                                             minSize=(5, 5),
                                                             flags=cv2.CASCADE_SCALE_IMAGE)
                if len(haar_detections2) > 0:  # HAAR Cascade 2 found faces
                    facefeatures = haar_detections2

            if len(facefeatures) == 0:
                haar_detections3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1,
                                                             minNeighbors=10,
                                                             minSize=(5, 5),
                                                             flags=cv2.CASCADE_SCALE_IMAGE)
                if len(haar_detections3) > 0:  # HAAR Cascade 3 found faces
                    facefeatures = haar_detections3

            if len(facefeatures) == 0:
                haar_detections4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1,
                                                             minNeighbors=10,
                                                             minSize=(5, 5),
                                                             flags=cv2.CASCADE_SCALE_IMAGE)
                if len(haar_detections4) > 0:  # HAAR Cascade 4 found faces
                    facefeatures = haar_detections4
                else:
                    facefeatures = []

        for (x, y, w, h) in facefeatures:
            print("Found a face!!")
            # cv2.imwrite('found.png', gray)
            # gray = gray[y:y+h, x:x+w]  # extract region of interest
            # cv2.imwrite('roi.png', gray)

            try:
                # gray = resize(gray)

                filenumber += 1
                cv2.imwrite("database_notresized//{}//{}.png".format(emotion, filenumber),
                            gray)
            except:
                continue

    return filenumber


total = 0
for emotion in emotions:
    filenum = detect_faces(emotion)
    total += filenum
    print("\n***> {} has {} images.\n".format(emotion, filenum))

print("***** Total Size of Dataset - {} *****\n".format(total))

# End the script.
end = time.clock()
print("***** Time elapsed: {} *****".format(end - start))

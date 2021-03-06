#!/usr/bin/env python3

"""Extraction of features, vectors, and faces, are held in the helper model."""

# Import packages.
import numpy as np
import cv2
import dlib
import math
import random
import glob
import time
from PIL import Image
from sklearn.decomposition import PCA

# My imports.
from sort_database.utils import *

# Set Face Detectors.
faceDet = cv2.CascadeClassifier("sort_database//" + HAAR)
faceDet2 = cv2.CascadeClassifier("sort_database//" + HAAR2)
faceDet3 = cv2.CascadeClassifier("sort_database//" + HAAR3)
faceDet4 = cv2.CascadeClassifier("sort_database//" + HAAR4)
faceDet5 = dlib.get_frontal_face_detector()  # dlib's face detector

# Build the required objects.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
predictor = dlib.shape_predictor("sort_database//" + PRED)

# Default sizes of training and testing sets (as ratios of the dataset)
training_set_size = 0.8
testing_set_size = 0.2


def convert_numpy(data):
    """Convert given data into numpy array."""
    return np.array(data)


def get_images(emotion):
    """Split dataset into 80 percent training set and 20 percent prediction."""
    files = glob.glob("sort_database//database//{}//*".format(emotion))
    random.shuffle(files)
    # training = files[:int(len(files) * training_set_size)]
    # prediction = files[-int(len(files) * testing_set_size):]
    # for testing
    training = files[:5]
    prediction = files[-5:]
    return training, prediction


def get_images2(emotion):
    """Split dataset into 80 percent training set and 20 percent prediction."""
    files = glob.glob("sort_database//database2//{}//*".format(emotion))
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * testing_set_size):]
    # for testing
    # training = files[:5]
    # prediction = files[-5:]
    return training, prediction


def get_face_opencv_vectors(img):
    """Get all the faces as OpenCV vectors."""
    haar_detections = []
    facefeatures = []

    # Use dlib to detect faces in the frame.
    detections = faceDet5(img, 1)

    if len(detections) > 0:  # dlib found faces
        for face in detections:
            x = face.left()
            y = face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            facefeatures.append([x, y, w, h])
    else:
        haar_detections = faceDet.detectMultiScale(img, scaleFactor=1.1,
                                                   minNeighbors=10,
                                                   minSize=(5, 5),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
        if len(haar_detections) > 0:  # HAAR Cascade found faces
            facefeatures = haar_detections

        if len(facefeatures) == 0:
            haar_detections2 = faceDet2.detectMultiScale(img, scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections2) > 0:  # HAAR Cascade 2 found faces
                facefeatures = haar_detections2

        if len(facefeatures) == 0:
            haar_detections3 = faceDet3.detectMultiScale(img, scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections3) > 0:  # HAAR Cascade 3 found faces
                facefeatures = haar_detections3

        if len(facefeatures) == 0:
            haar_detections4 = faceDet4.detectMultiScale(img, scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections4) > 0:  # HAAR Cascade 4 found faces
                facefeatures = haar_detections4
            # else:
            #     print("No face found")

    return facefeatures


def get_face_dlib_rects(img):
    """Get all the faces as dlib Rectangles."""
    haar_detections = []

    # Use dlib to detect faces in the frame.
    detections = faceDet5(img, 1)

    if len(detections) > 0:
        return detections
    else:
        haar_detections = faceDet.detectMultiScale(img, scaleFactor=1.1,
                                                   minNeighbors=10,
                                                   minSize=(5, 5),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)
        if len(haar_detections) > 0:  # HAAR Cascade found faces
            for (x, y, w, h) in haar_detections:
                dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                detections.append(dlib_rect)
                break

        if len(detections) == 0:
            haar_detections2 = faceDet2.detectMultiScale(img, scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections2) > 0:  # HAAR Cascade 2 found faces
                for (x, y, w, h) in haar_detections2:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break

        if len(detections) == 0:
            haar_detections3 = faceDet3.detectMultiScale(img, scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections3) > 0:  # HAAR Cascade 3 found faces
                for (x, y, w, h) in haar_detections3:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break

        if len(detections) == 0:
            haar_detections4 = faceDet4.detectMultiScale(img, scaleFactor=1.1,
                                                         minNeighbors=10,
                                                         minSize=(5, 5),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            if len(haar_detections4) > 0:  # HAAR Cascade 4 found faces
                for (x, y, w, h) in haar_detections4:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break
            # else:
                # print("No face found")

    return detections


def get_face_landmarks(img):
    """Get all the faces as flat landmark vectors."""
    detections = get_face_dlib_rects(img)

    # We may detect 0, 1 or many faces. Loop through each face detected.
    for i, j in enumerate(detections):
        # Draw facial landmarks with the predictor class.
        shape = predictor(img, j)
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

    if len(detections) == 0:
        return "error"
    return landmarks_vectorised


def get_sets_as_images(emotions):
    """Make the sets to test on."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        print("Obtaining images that represent ``{}''.".format(emotion))
        training, prediction = get_images2(emotion)

        print("Allocating {}. This directory contains {} images."
              .format(emotion, len(training)+len(prediction)))
        # Append data to training and prediction lists and label 0 to num.
        for item in training:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe_image = clahe.apply(gray)
            # clahe_image = cv2.equalizeHist(gray)
            # cv2.imwrite('clahe.png', clahe_image)
            check = get_face_dlib_rects(clahe_image)

            if len(check) > 0:
                training_data.append(clahe_image)
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe_image = clahe.apply(gray)
            # clahe_image = cv2.equalizeHist(gray)
            # cv2.imwrite('clahe.png', clahe_image)
            check = get_face_dlib_rects(clahe_image)

            if len(check) > 0:
                prediction_data.append(clahe_image)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def get_sets(emotions):
    """Make the feature vector sets to test on."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        print("Obtaining images that represent ``{}''.".format(emotion))
        training, prediction = get_images(emotion)

        print("***> Allocating {}. This directory contains {} images."
              .format(emotion, len(training)+len(prediction)))
        # Append data to training and prediction lists and label 0 to num.
        for item in training:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe_image = clahe.apply(gray)
            # clahe_image = cv2.equalizeHist(gray)
            # cv2.imwrite('clahe.png', clahe_image)
            facefeatures = get_face_landmarks(clahe_image)

            if (facefeatures != "error"):
                training_data.append(facefeatures)
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe_image = clahe.apply(gray)
            # clahe_image = cv2.equalizeHist(gray)
            # cv2.imwrite('clahe.png', clahe_image)
            facefeatures = get_face_landmarks(clahe_image)

            if (facefeatures != "error"):
                prediction_data.append(facefeatures)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def get_sets_pca(emotions):
    """Make the feature vector sets to test on."""
    X_train = X_test = y_train = y_test = []
    n_components = len(emotions)
    h = w = 380

    for emotion in emotions:
        print("Obtaining images that represent ``{}''.".format(emotion))
        training, prediction = get_images(emotion)

        for data in training:
            print(Image.open(data).shape)
            X_train.append(flatten_image(matrix_image(data)))
            y_train.append(emotions.index(emotion))
        for data in prediction:
            X_test.append(flatten_image(matrix_image(data)))
            y_test.append(emotions.index(emotion))

        X_train = convert_numpy(X_train)
        X_test = convert_numpy(X_test)

        print(X_train.shape, X_train.dtype)

        print("***> Allocating {}. This directory contains {} images."
              .format(emotion, len(training)+len(prediction)))

        print("Extracting the top {} eigenfaces from {} faces."
              .format(n_components, X_train.shape[0]))
        c0 = time.clock()
        pca = PCA(svd_solver='randomized', n_components=n_components).fit(X_train)
        print("Done in %0.3fs" % (time.clock() - c0))

        eigenfaces = pca.components_.reshape((n_components, h, w))

        print("Projecting the input data on the eigenfaces orthonormal basis.")

        t0 = time.clock()
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("done in %0.3fs" % (time.clock() - t0))

    return X_train_pca, X_test_pca, y_train, y_test

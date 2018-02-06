#!/usr/bin/env python3

"""
This is currently a demo of SVM and KNearest for emotion recognition.

Sample loads a dataset of images of faces from ''.
It then trains SVM and KNearest classfiers on the dataset and evaluates
their accuracy.
"""

import cv2 as cv2
import numpy as np
import glob
import random

from skimage.feature import hog
from sklearn import svm


el = ["neutral", "anger", "contempt", "disgust",
      "fear", "happy", "sadness", "surprise"]  # The emotion list


def get_images(emotion):
    """Split dataset into 80 percent training set and 20 percent prediction."""
    files = glob.glob("Database//{}//*".format(emotion))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


def sets(emotions):
    """Make the sets to test on."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        training, prediction = get_images(emotion)

        # Append data to training and prediction lists and label 0 to num.
        for item in training:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale

            fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualise=False)

            training_data.append(fd)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale

            fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualise=False)

            prediction_data.append(fd)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


X, y, pX, py = sets(el)

X = np.array(X)
y = np.array(y)

lin_clf = svm.LinearSVC()
print(lin_clf.fit(X, y))
print(lin_clf.score(X, y))

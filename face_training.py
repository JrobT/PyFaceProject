#!/usr/bin/env python3

"""Python script to train facial recognisers and test them on different sets.

I have defined 2 sets:
    -   Full Emotion List (8)
    -   The Emotion list without Fear, Sadness, and Neutral (5)

I have 3 classifiers:
    -   Fisher Faces
    -   Eigenfaces
    -   Local Binary Pattern Histograms
"""

# Import packages.
import cv2  # OpenCV
import os
import time
import multiprocessing
import numpy as np

# My imports.
import extraction_model as model
from constants import EMOTIONS_5, EMOTIONS_8


# Start the script.
start = time.clock()
sname = os.path.basename(__file__)  # The name of this script
print("\n{}: Program start. Clock: {}.".format(sname, start))

fisherface = cv2.face.FisherFaceRecognizer_create()  # Fisherface classifier
eigenface = cv2.face.EigenFaceRecognizer_create()  # Eigenface classifier
lbph = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns classifier


def run_fisher_recognizer(X_train, y_train, X_test, y_test):
    """Train the fisherface classifier."""
    print("\n***> Training fisherface classifier")
    print("Size of the training set is {} images.".format(len(y_train)))

    fisherface.train(X_train, np.asarray(y_train))

    print("***> Predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in X_test:
        pred, conf = fisherface.predict(image)
        if (pred == y_test[cnt]):
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1

    return ((correct / cnt) * 100)


def run_eigen_recognizer(X_train, y_train, X_test, y_test):
    """Train the eigenface classifier."""
    print("\n***>: Training eigenface classifier")
    print("Size of the training set is {} images.".format(len(y_train)))

    eigenface.train(X_train, np.asarray(y_train))

    print("***> Predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in X_test:
        pred, conf = eigenface.predict(image)
        if (pred == y_test[cnt]):
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1

    return ((correct / cnt) * 100)


def run_lbph_recognizer(X_train, y_train, X_test, y_test):
    """Train the local binary pattern classifier."""
    print("\n***>: Training local binary pattern classifier")
    print("Size of the training set is {} images.".format(len(y_train)))

    lbph.train(X_train, np.asarray(y_train))

    print("***> Predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in X_test:
        pred, conf = lbph.predict(image)
        if (pred == y_test[cnt]):
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1

    return ((correct / cnt) * 100)


X_train, y_train, X_test, y_test = model.get_sets_as_images(EMOTIONS_8)
X_train1, y_train1, X_test1, y_test1 = model.get_sets_as_images(EMOTIONS_5)

metascore1 = []
metascore2 = []
metascore3 = []
metascore4 = []
metascore5 = []
metascore6 = []


def threadme1():
    """Produce the results on thread 1."""
    for i in range(0, 5):
        f = run_fisher_recognizer(X_train, y_train, X_test, y_test)
        print("***> Fisherfaces/8 Emotions got {} percent correct".format(f))
        metascore1.append(f)


def threadme2():
    """Produce the results on thread 2."""
    for i in range(0, 5):
        f1 = run_fisher_recognizer(X_train1, y_train1, X_test1, y_test1)
        print("***> Fisherfaces/5 Emotions got {} percent correct".format(f1))
        metascore2.append(f1)


def threadme3():
    """Produce the results on thread 3."""
    for i in range(0, 5):
        e = run_eigen_recognizer(X_train, y_train, X_test, y_test)
        print("***> Eigenfaces/8 Emotions got {} percent correct".format(e))
        metascore3.append(e)


def threadme4():
    """Produce the results on thread 4."""
    for i in range(0, 5):
        e1 = run_eigen_recognizer(X_train1, y_train1, X_test1, y_test1)
        print("***> Eigenfaces/5 Emotions got {} percent correct".format(e1))
        metascore4.append(e1)


def threadme5():
    """Produce the results on thread 5."""
    for i in range(0, 5):
        l = run_lbph_recognizer(X_train, y_train, X_test, y_test)
        print("***> Local Binary Pattern Histograms/8 Emotions got {} percent correct".format(l))
        metascore5.append(l)


def threadme6():
    """Produce the results on thread 6."""
    for i in range(0, 5):
        l1 = run_lbph_recognizer(X_train1, y_train1, X_test1, y_test1)
        print("***> Local Binary Pattern Histograms/5 Emotions got {} percent correct".format(l1))
        metascore6.append(l1)


# Run tasks using processes.
processes = [multiprocessing.Process(target=threadme1()),
             multiprocessing.Process(target=threadme2()),
             multiprocessing.Process(target=threadme3()),
             multiprocessing.Process(target=threadme4()),
             multiprocessing.Process(target=threadme5()),
             multiprocessing.Process(target=threadme6())]
[process.start() for process in processes]
[process.join() for process in processes]

print("\n***> Final score for Fisherfaces/8 Emotions: {} percent correct."
      .format(np.mean(metascore1)))
print("***> Final score for Fisherfaces/5 Emotions: {} percent correct."
      .format(np.mean(metascore2)))
print("***> Final score for Eigenfaces/8 Emotions: {} percent correct."
      .format(np.mean(metascore3)))
print("***> Final score for Eigenfaces/5 Emotions: {} percent correct."
      .format(np.mean(metascore4)))
print("***> Final score for Local Binary Histograms/8 Emotions: {} percent correct."
      .format(np.mean(metascore5)))
print("***> Final score for Local Binary Histograms/5 Emotions: {} percent correct."
      .format(np.mean(metascore6)))

# End the script.
end = time.clock()
print("\n{}: Program end. Time elapsed: {}.".format(sname, end - start))

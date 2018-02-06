#!/usr/bin/env python3

"""Python script to evaluate emotional recognition using a SVM classifier.

In essence, this script will produce accuracy scores for different methods of
classifier when performing emotional recongition on a dataset. From my tests
I show a SVM to be the best way of achieving this that I have tested within
this scope, and so this script may be extended into the music player component
of the project to demonstrate a use of this technology (unless I can adapt a
CNN approach).

I use OpenCV and dlib to build the datasets, and the sklearn library to
obtain an implementation of a support vector machine (SVM). Here, I focus on
building 3 SVMs with different kernels and will evaluate their performance
based on the combined dataset I have created previously. Currently, each SVM is
trained using a training set made from a random sample of 80% of the
dataset and then tested on the remaining 20%.

I am evaluating the performance on a grayscale set of images and the facial
expression labels declared.. The kernels I will be testing are linear,
radial basis function, and polynomial kernels.
"""

import numpy as np  # SVC methods will accept numpy arrays

# Import SVC. This is an implementation of an SVM based on libsvm.
from sklearn.svm import SVC  # default tolerance=1e-3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import cv2  # OpenCV
import dlib
import glob
import random
import math
import os
import time  # To speed test


# Start the script.
script_name = os.path.basename(__file__)  # The name of this script

print("\n\n{}: Program starts...".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate.

el = ["neutral", "anger", "contempt", "disgust",
      "fear", "happy", "sadness", "surprise"]  # The emotion list

# Build the required objects.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("FaceLandmarks.dat")  # file must be in dir

""" Set the classifier to a support vector machine with a linear kernel """
linearClf = SVC(kernel='linear', probability=True)
svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("", LinearSVC)
))

""" Set the classifier to a support vector machine with a radial basis
function kernel """
rbfClf = SVC(kernel='rbf', probability=True)

""" Set the classifier to a support vector machine with a polynomial kernel """
polyClf = SVC(kernel='poly', probability=True)

data = {}  # Make dictionary for all data point values

# Default sizes of training and testing sets (as ratios of the dataset)
training_set_size = 0.9
testing_set_size = 0.1


def get_images(emotion):
    """Split dataset into 90 percent training set and 10 percent prediction."""
    files = glob.glob("Database//{}//*".format(emotion))
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * testing_set_size):]
    # training = files[:int(5)]
    # prediction = files[-int(5):]
    return training, prediction


def get_landmarks(image):
    """Overlay the predictor and turn into features for the classifier.

    I am sourcing the coordinates of the landmarks. This is a feature because
    where a landmarks is could be indicative different AUs in FACS for example,
    which indicate emotion. The landmark's location is only important relative
    to positioning on the face, therefore these need to be normalised between
    0 and 1. However strict normalisation would confuse two features that are
    direct opposites. Instead I use centre point to create vectors around the
    face. Finally, an adjustment needs to be made for tilted or miss-aligned
    faces.
    """
    # Use dlib's detector to detect faces in the frame (not HAAR Cascades).
    detections = detector(image, 1)

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

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


def sets(emotions):
    """Make the sets to test on."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        print("{}: Allocating {}...".format(script_name, emotion))

        training, prediction = get_images(emotion)

        # Append data to training and prediction lists and label 0 to num.
        for item in training:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale

            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)

            if data['landmarks_vectorised'] == "error":
                print("""{}: ERROR CODE. No face detected on this
                      data point.""".format(script_name))
            else:
                training_data.append(data['landmarks_vectorised'])
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale

            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)

            if data['landmarks_vectorised'] == "error":
                print("""{}: ERROR CODE. No face detected on this
                      data point.""".format(script_name))
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


""" Test the classifiers! """
# Array's to hold each score.
linear_scores = []
rbf_scores = []
poly_scores = []

for i in range(0, 5):  # I'm choosing to test 5 times
    print("{}: Making training and prediction sets...".format(script_name))
    tdata, tlabels, pdata, plabels = sets(el)

    # Change to numpy arrays as classifier expects them in this format
    nptdata = np.array(tdata)
    nppdata = np.array(pdata)

    # Train the support vector machine with linear kernel.
    print("{}: Training SVM with linear kernel...".format(script_name))
    linearClf.fit(nptdata, tlabels)

    # Train the support vector machine with radial basis function kernel.
    print("{}: Training SVM with RBF kernel...".format(script_name))
    rbfClf.fit(nptdata, tlabels)

    # Train the support vector machine with polynomial kernel.
    print("{}: Training SVM with polynomial kernel...".format(script_name))
    polyClf.fit(nptdata, tlabels)

    # Get the score for the classfiers (percent it got correct).
    print("""{}: Scoring the classifiers on the
          prediction set...""".format(script_name))
    linear_score = linearClf.score(nppdata, plabels)
    rbf_score = rbfClf.score(nppdata, plabels)
    poly_score = polyClf.score(nppdata, plabels)

    # Output the results of this round of testing.
    print("\n{}: Test round {:d} - ".format(script_name, (i+1)))
    print("""{}: Support Vector Machine with a Linear kernel got
          {:.2f} percent correct!""".format(script_name, (linear_score*100)))
    print("""{}: Support Vector Machine with a Radial Basis Function kernel
          got {:.2f} percent correct!""".format(script_name, (rbf_score*100)))
    print("""{}: Support Vector Machine with a Polynomial kernel got
          {:.2f} percent correct!""".format(script_name, (poly_score*100)))

    # Append this rounds scores to the respective metascore array.
    linear_scores.append(linear_score)
    rbf_scores.append(rbf_score)
    poly_scores.append(poly_score)

print("""\n\n{}: Average accuracy score for Support Vector Machine with
      Linear kernel - {:.2f}%""".format(script_name,
                                        (np.mean(linear_scores)*100)))
print("""{}: Average accuracy score for Support Vector Machine with Radial
      Basis Function kernel - {:.2f}%""".format(script_name,
                                                (np.mean(rbf_scores)*100)))
print("""{}: Average score for Support Vector Machine with Polynomial
      kernel - {:.2f}%""".format(script_name, (np.mean(poly_scores)*100)))

# End the script.
end = time.clock()
print("""\n{}: Program end. Time elapsed:
      {:.5f}.""".format(script_name, end - start))

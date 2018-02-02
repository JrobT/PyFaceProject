#!/usr/bin/env python3

"""
Python script to evaluate facial expression recognition using SVM classifiers.

In essence, this script will produce accuracy scores for different methods of
classifier when performing emotional recognition on a dataset. From my tests
I show a SVM to be the best way of achieving this that I have tested within
this scope, and so this script may be extended into the music player component
of the project to demonstrate a use of this technology (unless I can adapt a
CNN approach).

From ``Hands-On Machine Learning with Scikit-Learn & TensorFlow'' -
    A Support Vector Machine (SVM) is a very powerful and versatile Machine
    Learning model, capable of performing linear or nonlinear classification,
    regression, and even outlier detection.''

I use OpenCV and dlib to build and manage the datasets, and the sklearn library
to obtain an implementation of a support vector machine (SVM). Here, I focus on
building 3 SVMs with different kernels and will evaluate their performance
based on the combined dataset I have created previously. Currently, each SVM is
trained using a training set made from a random sample of 80% of the
dataset and then tested on the remaining 20%. Facial landmark vectors and
eigenfaces vectors are both used for feature extraction.

I am evaluating the performance on a grayscale set of images and the facial
expression labels declared. The kernels I will be testing are linear,
radial basis function, and polynomial kernels.

After allocation of the dataset (will take a while, due to sample size), I test
each method and label its output.
"""

import numpy as np  # SVC methods will accept numpy arrays
import matplotlib.pyplot as plt  # Plot ROC curve
import cv2
import dlib
import glob
import math
import os
import random
import time

# Importing Scikit-Learn, preprocessors and constants.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from imutils import face_utils
from constants import HAAR, HAAR2, HAAR3, HAAR4, PRED, EMOTIONS_8, EMOTIONS_5


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: Beginning Support Vector Machine tests...".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate

# Set Face Detectors.
faceDet = cv2.CascadeClassifier(HAAR)
faceDet2 = cv2.CascadeClassifier(HAAR2)
faceDet3 = cv2.CascadeClassifier(HAAR3)
faceDet4 = cv2.CascadeClassifier(HAAR4)
faceDet5 = dlib.get_frontal_face_detector()  # dlib's face detector

# Build the required objects.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
predictor = dlib.shape_predictor(PRED)  # file must be in dir

data = {}  # Make dictionary for all data point values

# Default sizes of training and testing sets (as ratios of the dataset)
training_set_size = 0.8
testing_set_size = 0.2


def get_images(emotion):
    """Split dataset into 80 percent training set and 20 percent prediction."""
    files = glob.glob("sort_database//database//{}//*".format(emotion))
    random.shuffle(files)
    training = files[:int(len(files) * training_set_size)]
    prediction = files[-int(len(files) * testing_set_size):]
    # for testing
    # training = files[:5]
    # prediction = files[-5:]
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
    faces. The image's feature vector is stored for the training data.
    """
    # Use detectors to detect faces in the frame.
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
    if len(detections) < 1:  # weird, because dataset has been run through with detectors!
        data['landmarks_vectorised'] = "error"


# def create_data(emotions):
#     """."""
#     X_train = []
#     X_test = []
#     y_train = []
#     y_test = []
#
#     for emotion in emotions:
#         print("Obtaining images that represent ``{}''.".format(emotion))
#         training, prediction = get_images(emotion)
#
#         print("Allocating {}. This directory contains {} images."
#               .format(emotion, len(training)+len(prediction)))
#
#     for item in training:
#         img = cv2.imread(item, 0)
#         gray = np.array(img, 'uint8')
#
#         # Look for faces.
#         face = faceDet.detectMultiScale(gray)
#         face2 = faceDet2.detectMultiScale(gray)
#         face3 = faceDet3.detectMultiScale(gray)
#         face4 = faceDet4.detectMultiScale(gray)
#         face5 = faceDet5(gray, 1)
#
#         if len(face) == 1:
#             facefeatures = face
#         elif len(face2) == 1:
#             facefeatures = face2
#         elif len(face3) == 1:
#             facefeatures = face3
#         elif len(face4) == 1:
#             facefeatures = face4
#         elif len(face5) == 1:
#             facefeatures = [face_utils.rect_to_bb(face5[0])]
#         else:
#             facefeatures = ""
#
#         for (x, y, w, h) in facefeatures:
#             gray = gray[y:y+h, x:x+w]
#             X_train.append(gray)
#             y_train.append(emotions.index(emotion))
#
#     for item in prediction:
#         frame = cv2.imread(item)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale
#
#         # Look for faces.
#         face = faceDet.detectMultiScale(gray, scaleFactor=1.1,
#                                         minNeighbors=10, minSize=(5, 5),
#                                         flags=cv2.CASCADE_SCALE_IMAGE)
#         face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1,
#                                           minNeighbors=10, minSize=(5, 5),
#                                           flags=cv2.CASCADE_SCALE_IMAGE)
#         face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1,
#                                           minNeighbors=10, minSize=(5, 5),
#                                           flags=cv2.CASCADE_SCALE_IMAGE)
#         face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1,
#                                           minNeighbors=10, minSize=(5, 5),
#                                           flags=cv2.CASCADE_SCALE_IMAGE)
#         face5 = faceDet5(gray, 1)
#
#         if len(face) == 1:
#             facefeatures = face
#         elif len(face2) == 1:
#             facefeatures = face2
#         elif len(face3) == 1:
#             facefeatures = face3
#         elif len(face4) == 1:
#             facefeatures = face4
#         elif len(face5) == 1:
#             facefeatures = [face_utils.rect_to_bb(face5[0])]
#         else:
#             facefeatures = ""
#
#         for (x, y, w, h) in facefeatures:
#             gray = gray[y:y+h, x:x+w]
#             X_test.append(gray)
#             y_test.append(emotions.index(emotion))
#
#     return X_train, X_test, y_train, y_test


# def get_pca(emotions):
#     """."""
#     X_train, X_test, y_train, y_test = create_data(emotions)
#
#     n_components = 10
#     # nsamples, nx, ny = X_train.shape
#     # X_train = X_train.reshape((nsamples, nx*ny))
#     # nsamples, nx, ny = X_test.shape
#     # X_test = X_test.reshape((nsamples, nx*ny))
#     X_train = np.array(X_train)
#     pca = PCA(n_components=n_components, svd_solver='randomized',
#               whiten=True).fit(X_train)
#
#     # project the input data on the eigenfaces orthonormal basis
#     X_train_pca = pca.transform(X_train)
#     X_test_pca = pca.transform(X_test)
#
#     return X_train_pca, X_test_pca, y_train, y_test


def sets(emotions):
    """Make the sets to test on."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        print("Obtaining images that represent ``{}''.".format(emotion))
        training, prediction = get_images(emotion)

        print("Allocating {}. This directory contains {} images."
              .format(emotion, len(training)+len(prediction)))
        # Append data to training and prediction lists and label 0 to num.
        for item in training:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)

            # if data['landmarks_vectorised'] == "error":
            #     # print("""{}: ERROR CODE. No face detected on this
            #     #    data point.""".format(script_name))  # Doesn't happen
            # else:
            if data['landmarks_vectorised'] != "error":
                training_data.append(data['landmarks_vectorised'])
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)  # Open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)

            # if data['landmarks_vectorised'] == "error":
            #     # print("""{}: ERROR CODE. No face detected on this
            #     #    data point.""".format(script_name))  # Doesn't happen
            # else:
            if data['landmarks_vectorised'] != "error":
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def build_roc_curve(n_classes, y_score, y_test, title):
    """Build a ROC curve for a multiclass problem."""
    # Compute ROC curve and ROC area for each class.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area.
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve.
    fig = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass Receiver Operating Characteristic Curve for {}'.format(title))
    plt.legend(loc="lower right")
    fig.savefig('roc_curves/{}'.format(title))


""" Test the classifiers! """

# Array's to hold all the acc. data to average.
# Linear Kernel.
linear_scores = []
linear_scores1 = []
linear_scores_pca = []
linear_fit_times = []
linear_fit_times_pca = []
linear_fit_times1 = []
linear_score_times = []
linear_score_times_pca = []
linear_score_times1 = []
# Radial Basis Function Kernal.
rbf_scores = []
rbf_scores1 = []
rbf_fit_times = []
rbf_fit_times1 = []
rbf_score_times = []
rbf_score_times1 = []
# Polynomial Kernel.
poly_scores = []
poly_scores1 = []
poly_fit_times = []
poly_fit_times1 = []
poly_score_times = []
poly_score_times1 = []
# Timing variables.
fit_time_start = 0
fit_time_stop = 0
score_time_start = 0
score_time_stop = 0
fit_time = 0
score_time = 0

# Build classifiers.
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge"))
])
rbf_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svc_clf", SVC(kernel="rbf", gamma=5, C=0.001, probability=True))
))
poly_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svc_clf", SVC(kernel="poly", degree=3, coef0=1, C=5, probability=True))
))

for i in range(0, 5):  # 5 testing runs
    print("\nROUND {}\n".format(i+1))
    tdata, tlabels, pdata, plabels = sets(EMOTIONS_8)
    tdata1, tlabels1, pdata1, plabels1 = sets(EMOTIONS_5)
    # tdatapca, pdatapca, tlabelspca, plabelspca = get_pca(el)
    # tdatapca, pdatapca, tlabelspca, plabelspca = get_pca(el1)

    # Change to numpy arrays as classifier expects them in this format.
    nptdata = np.array(tdata)
    nppdata = np.array(pdata)
    nptdata1 = np.array(tdata1)
    nppdata1 = np.array(pdata1)

    # Binarize the output.
    y = label_binarize(plabels, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    y1 = label_binarize(plabels1, classes=[1, 2, 3, 4, 5])
    n_classes = y.shape[1]
    n_classes1 = y1.shape[1]

    """ LINEAR KERNEL // LIST 1 // LANDMARKS """
    # Probability of one class vs. the rest.
    lin_svm_clf = OneVsRestClassifier(svm_clf)

    # BINARY LABELS??

    # Train the support vector machine.
    print("\nTraining SVM with LINEAR kernel & list 1.")
    fit_time_start = time.clock()
    y_score = lin_svm_clf.fit(nptdata, tlabels).decision_function(nppdata)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    # Get the score for the classifier (percent it got correct).
    print("Scoring the classifier on the prediction list 1.")
    score_time_start = time.clock()
    linear_score = lin_svm_clf.score(nppdata, plabels)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    # Multiclass ROC curve generator.
    build_roc_curve(n_classes, y_score, y, "Linear Kernel (8_{})".format(i+1))

    # Append this rounds scores and times to the respective metascore array.
    linear_scores.append(linear_score)
    linear_fit_times.append(fit_time)
    linear_score_times.append(score_time)

    # """ LINEAR KERNEL // LIST 1 // PCA """
    #
    # # Train the support vector machine with linear kernel.
    # print("Training SVM with linear kernel, list 1 & PCA.")
    # fit_time_start = time.clock()
    # svm_clf.fit(tdatapca, tlabelspca)
    # fit_time_stop = time.clock()
    # fit_time = fit_time_stop-fit_time_start
    #
    # # Get the score for the classfier (percent it got correct).
    # print("Scoring the classfier on the prediction set 1.")
    # score_time_start = time.clock()
    # linear_score = svm_clf.score(pdatapca, plabelspca)
    # score_time_stop = time.clock()
    # score_time = score_time_stop-score_time_start
    #
    # # Append this rounds scores and times to the respective metascore array.
    # linear_scores_pca.append(linear_score)
    # linear_fit_times_pca.append(fit_time)
    # linear_score_times_pca.append(score_time)

    """ LINEAR KERNEL // LIST 2 // LANDMARKS """
    lin_svm_clf = OneVsRestClassifier(svm_clf)

    print("\nTraining SVM with LINEAR kernel & list 2.")
    fit_time_start = time.clock()
    y_score = lin_svm_clf.fit(nptdata1, tlabels1).decision_function(nppdata1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    linear_score = lin_svm_clf.score(nppdata1, plabels1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    build_roc_curve(n_classes1, y_score, y1, "Linear Kernel (5_{})".format(i+1))

    linear_scores1.append(linear_score)
    linear_fit_times1.append(fit_time)
    linear_score_times1.append(score_time)

    """ RBF KERNEL // LIST 1 // LANDMARKS """
    rbf_clf = OneVsRestClassifier(rbf_svm_clf)

    print("\nTraining SVM with RBF kernel & list 1.")
    fit_time_start = time.clock()
    y_score = rbf_clf.fit(nptdata, tlabels).decision_function(nppdata)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 1.")
    score_time_start = time.clock()
    rbf_score = rbf_clf.score(nppdata, plabels)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    build_roc_curve(n_classes, y_score, y, "RBF Kernel (8_{})".format(i+1))

    rbf_scores.append(rbf_score)
    rbf_fit_times.append(fit_time)
    rbf_score_times.append(score_time)

    """ RBF KERNEL // LIST 2 // LANDMARKS """
    rbf_clf = OneVsRestClassifier(rbf_svm_clf)

    print("Training SVM with RBF kernel & list 2.")
    fit_time_start = time.clock()
    y_score = rbf_clf.fit(nptdata1, tlabels1).decision_function(nppdata1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    rbf_score1 = rbf_clf.score(nppdata1, plabels1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    build_roc_curve(n_classes1, y_score, y1, "RBF Kernel (5_{})".format(i+1))

    rbf_scores1.append(rbf_score1)
    rbf_fit_times1.append(fit_time)
    rbf_score_times1.append(score_time)

    """ POLY KERNEL // LIST 1 // LANDMARKS """
    poly_clf = OneVsRestClassifier(poly_svm_clf)

    print("Training SVM with POLYNOMIAL kernel & list 1.")
    fit_time_start = time.clock()
    y_score = poly_clf.fit(nptdata, tlabels).decision_function(nppdata)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 1.")
    score_time_start = time.clock()
    poly_score = poly_clf.score(nppdata, plabels)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    build_roc_curve(n_classes, y_score, y, "Polynomial Kernel (8_{})".format(i+1))

    poly_scores.append(poly_score)
    poly_fit_times.append(fit_time)
    poly_score_times.append(score_time)

    """ POLY KERNEL // LIST 2 """
    poly_clf = OneVsRestClassifier(poly_svm_clf)

    print("Training SVM with POLYNOMIAL kernel & list 2.")
    fit_time_start = time.clock()
    y_score = poly_clf.fit(nptdata1, tlabels1).decision_function(nppdata1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    poly_score = poly_clf.score(nppdata1, plabels1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    build_roc_curve(n_classes1, y_score, y1, "Polynomial Kernel (5_{})".format(i+1))

    poly_scores1.append(poly_score)
    poly_fit_times1.append(fit_time)
    poly_score_times1.append(score_time)

    """ END OF THE ROUND """

    print("\n\n{}: Test round {} - ".format(script_name, (i+1)))
    print("\nSVM, LINEAR, Landmarks, List 1:")
    print("> {:.2f} percent correct.".format(linear_scores[i]*100))
    print("> Fit time {}".format(linear_fit_times[i]))
    print("> Test time {}".format(linear_score_times[i]))
    print("\nSVM, LINEAR, Landmarks, List 2:")
    print("> {:.2f} percent correct.".format(linear_scores1[i]*100))
    print("> Fit time {}".format(linear_fit_times1[i]))
    print("> Test time {}".format(linear_score_times1[i]))
    # print("\nSVM, Linear, PCA, List 1:")
    # print("> {:.2f} percent correct.".format(linear_scores_pca[i]*100))
    # print("> Fit time {}".format(linear_fit_times_pca[i]))
    # print("> Test time {}".format(linear_score_times_pca[i]))
    print("\nSVM, RBF, Landmarks, List 1:")
    print("> {:.2f} percent correct.".format(rbf_scores[i]*100))
    print("> Fit time {}".format(rbf_fit_times[i]))
    print("> Test time {}".format(rbf_score_times[i]))
    print("\nSVM, RBF, Landmarks, List 2:")
    print("> {:.2f} percent correct.".format(rbf_scores1[i]*100))
    print("> Fit time {}".format(rbf_fit_times1[i]))
    print("> Test time {}".format(rbf_score_times1[i]))
    print("\nSVM, POLYNOMIAL, Landmarks, List 1:")
    print("> {:.2f} percent correct.".format(poly_scores[i]*100))
    print("> Fit time {}".format(poly_fit_times[i]))
    print("> Test time {}".format(poly_score_times[i]))
    print("\nSVM, POLYNOMIAL, Landmarks, List 2:")
    print("> {:.2f} percent correct.".format(poly_scores1[i]*100))
    print("> Fit time {}".format(poly_fit_times1[i]))
    print("> Test time {}".format(poly_score_times1[i]))

""" END OF THE TEST """

print("\n\nAverages for SVM with Linear kernel / set 1 - {:.2f}%, {}, {}.".format((np.mean(linear_scores)*100),
                                                                                  (np.mean(linear_fit_times)),
                                                                                  (np.mean(linear_score_times))))
print("Averages for SVM with Linear kernel / set 2 - {:.2f}%, {}, {}.".format((np.mean(linear_scores1)*100),
                                                                              (np.mean(linear_fit_times1)),
                                                                              (np.mean(linear_score_times1))))
print("\nAverages for SVM with RBF kernel / set 1 - {:.2f}%, {}, {}.".format((np.mean(rbf_scores)*100),
                                                                           (np.mean(rbf_fit_times)),
                                                                           (np.mean(rbf_score_times))))
print("Averages for SVM with RBF kernel / set 2 - {:.2f}%, {}, {}.".format((np.mean(rbf_scores1)*100),
                                                                           (np.mean(rbf_fit_times1)),
                                                                           (np.mean(rbf_score_times1))))
print("\nAverages for SVM with POLY kernel / set 1 - {:.2f}%, {}, {}.".format((np.mean(poly_scores)*100),
                                                                            (np.mean(poly_fit_times)),
                                                                            (np.mean(poly_score_times))))
print("Averages for SVM with POLY kernel / set 2 - {:.2f}%, {}, {}.".format((np.mean(poly_scores1)*100),
                                                                            (np.mean(poly_fit_times1)),
                                                                            (np.mean(poly_score_times1))))

# End the script.
end = time.clock()
print("""\n{}: Program end. Time elapsed:
      {:.5f}.""".format(script_name, end - start))

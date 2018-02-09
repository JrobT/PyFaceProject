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
import os
import time

# Trying PCA.
# from imutils import face_utils
# from sklearn.decomposition import PCA

# Importing Scikit-Learn preprocessors.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

# My imports.
import extraction_model as model
from constants import EMOTIONS_8, EMOTIONS_5


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: Beginning Support Vector Machine tests...\n".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate


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
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass Receiver Operating Characteristic Curve for {}'
              ''.format(title))
    plt.legend(loc="lower right")
    fig.savefig('roc_curves_reports/{}'.format(title))


""" Test the classifiers! """

# Array's to hold all the acc. data to average.
# Linear Kernel.
linear_scores = []
linear_scores1 = []
linear_fit_times = []
linear_fit_times1 = []
linear_score_times = []
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
    tdata, tlabels, pdata, plabels = model.get_sets(EMOTIONS_8)
    tdata1, tlabels1, pdata1, plabels1 = model.get_sets(EMOTIONS_5)

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

    # Train the support vector machine.
    print("***> Training SVM with LINEAR kernel & list 1.")
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

    name = "Linear Kernel (8_{})".format(i+1)
    # Multiclass ROC curve generator.
    build_roc_curve(n_classes, y_score, y, name)
    # Classification Report.
    model.produce_report(lin_svm_clf, nppdata, plabels, n_classes, name)

    # Append this rounds scores and times to the respective metascore array.
    linear_scores.append(linear_score)
    linear_fit_times.append(fit_time)
    linear_score_times.append(score_time)

    """ LINEAR KERNEL // LIST 2 // LANDMARKS """
    lin_svm_clf = OneVsRestClassifier(svm_clf)

    print("***> Training SVM with LINEAR kernel & list 2.")
    fit_time_start = time.clock()
    y_score = lin_svm_clf.fit(nptdata1, tlabels1).decision_function(nppdata1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    linear_score = lin_svm_clf.score(nppdata1, plabels1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "Linear Kernel (5_{})".format(i+1)
    build_roc_curve(n_classes1, y_score, y1, name)
    model.produce_report(lin_svm_clf, nppdata1, plabels1, n_classes1, name)

    linear_scores1.append(linear_score)
    linear_fit_times1.append(fit_time)
    linear_score_times1.append(score_time)

    """ RBF KERNEL // LIST 1 // LANDMARKS """
    rbf_clf = OneVsRestClassifier(rbf_svm_clf)

    print("***> Training SVM with RBF kernel & list 1.")
    fit_time_start = time.clock()
    y_score = rbf_clf.fit(nptdata, tlabels).decision_function(nppdata)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 1.")
    score_time_start = time.clock()
    rbf_score = rbf_clf.score(nppdata, plabels)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "RBF Kernel (8_{})".format(i+1)
    build_roc_curve(n_classes, y_score, y, name)
    model.produce_report(rbf_clf, nppdata, plabels, n_classes, name)

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
    rbf_score = rbf_clf.score(nppdata1, plabels1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "RBF Kernel (5_{})".format(i+1)
    build_roc_curve(n_classes1, y_score, y1, name)
    model.produce_report(rbf_clf, nppdata1, plabels1, n_classes1, name)

    rbf_scores1.append(rbf_score)
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

    name = "Polynomial Kernel (8_{})".format(i+1)
    build_roc_curve(n_classes, y_score, y, name)
    model.produce_report(poly_clf, nppdata, plabels, n_classes, name)

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

    name = "Polynomial Kernel (8_{})".format(i+1)
    build_roc_curve(n_classes1, y_score, y1, name)
    model.produce_report(poly_clf, nppdata1, plabels1, n_classes1, name)

    poly_scores1.append(poly_score)
    poly_fit_times1.append(fit_time)
    poly_score_times1.append(score_time)

    """ END OF THE ROUND """

    print("\n{}: Test round {} - ".format(script_name, (i+1)))
    print("\nSVM, LINEAR, Landmarks, List 1:")
    print("> {:.2f} percent correct.".format(linear_scores[i]*100))
    print("> Fit time {}".format(linear_fit_times[i]))
    print("> Test time {}".format(linear_score_times[i]))
    print("\nSVM, LINEAR, Landmarks, List 2:")
    print("> {:.2f} percent correct.".format(linear_scores1[i]*100))
    print("> Fit time {}".format(linear_fit_times1[i]))
    print("> Test time {}".format(linear_score_times1[i]))
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

print("\nAverages for SVM with Linear kernel / set 1 - {:.2f}%, {}, {}."
      .format((np.mean(linear_scores)*100),
              (np.mean(linear_fit_times)),
              (np.mean(linear_score_times))))
print("Averages for SVM with Linear kernel / set 2 - {:.2f}%, {}, {}."
      .format((np.mean(linear_scores1)*100),
              (np.mean(linear_fit_times1)),
              (np.mean(linear_score_times1))))
print("\nAverages for SVM with RBF kernel / set 1 - {:.2f}%, {}, {}."
      .format((np.mean(rbf_scores)*100),
              (np.mean(rbf_fit_times)),
              (np.mean(rbf_score_times))))
print("Averages for SVM with RBF kernel / set 2 - {:.2f}%, {}, {}."
      .format((np.mean(rbf_scores1)*100),
              (np.mean(rbf_fit_times1)),
              (np.mean(rbf_score_times1))))
print("\nAverages for SVM with POLY kernel / set 1 - {:.2f}%, {}, {}."
      .format((np.mean(poly_scores)*100),
              (np.mean(poly_fit_times)),
              (np.mean(poly_score_times))))
print("Averages for SVM with POLY kernel / set 2 - {:.2f}%, {}, {}."
      .format((np.mean(poly_scores1)*100),
              (np.mean(poly_fit_times1)),
              (np.mean(poly_score_times1))))

# End the script.
end = time.clock()
print("""\n{}: Program end. Time elapsed:
      {:.5f}.""".format(script_name, end - start))

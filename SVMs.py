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

# Import packages.
import numpy as np  # SVC methods will accept numpy arrays
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
from sklearn.model_selection import GridSearchCV

# My imports.
import extraction_model as exmodel
import evaluation_model as evmodel
from constants import EMOTIONS_8, EMOTIONS_5


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: Beginning Support Vector Machine tests...\n".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate


def rbf_param_selection(X, y):
    """."""
    Cs = [0.01, 0.1, 1, 5, 10]
    gammas = [0.01, 0.1, 1, 5, 10]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def linear_param_selection(X, y):
    """."""
    Cs = [0.01, 0.1, 1, 5, 10]
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def poly_param_selection(X, y):
    """."""
    Cs = [0.01, 0.1, 1, 5, 10]
    gammas = [0.01, 0.1, 1, 5, 10]
    degreeValues = [1, 2, 3]
    coef0Values = [0, 1, 2]
    param_grid = {'C': Cs, 'gamma': gammas,
                  'degree': degreeValues, 'coef0': coef0Values}
    grid_search = GridSearchCV(SVC(kernel='poly'), param_grid)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


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

X_train1, y_train1, X_test1, y_test1 = exmodel.get_sets(EMOTIONS_5)
X = X_train1 + X_test1
y = y_train1 + y_test1
print(rbf_param_selection(X, y))
print(linear_param_selection(X, y))
print(poly_param_selection(X, y))

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
# GridSearchCV found C=0.01 to be optimal parameter.
lin_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=0.01, loss="hinge"))
])
# GridSearchCV found C=5, gamma=0.01 to be optimal parameters.
rbf_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svc_clf", SVC(kernel="rbf", C=5, gamma=0.01, probability=True))
))
poly_svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("svc_clf", SVC(kernel="poly", C=1, probability=True))
))

for i in range(0, 5):  # 5 testing runs
    print("\nROUND {}\n".format(i+1))
    X_train, y_train, X_test, y_test = exmodel.get_sets(EMOTIONS_8)
    X_train1, y_train1, X_test1, y_test1 = exmodel.get_sets(EMOTIONS_5)

    # Change to numpy arrays as classifier expects them in this format.
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train1 = np.array(X_train1)
    X_test1 = np.array(X_test1)

    # Binarize the output.
    y = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    y1 = label_binarize(y_test1, classes=[1, 2, 3, 4, 5])
    n_classes = y.shape[1]
    n_classes1 = y1.shape[1]

    """ LINEAR KERNEL // LIST 1 // LANDMARKS """
    # Probability of one class vs. the rest.
    svm_clf = OneVsRestClassifier(lin_svm_clf)

    # Train the support vector machine.
    print("***> Training SVM with LINEAR kernel & list 1.")
    fit_time_start = time.clock()
    y_pred = svm_clf.fit(X_train, y_train).predict(X_test)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    # Get the score for the classifier (percent it got correct).
    print("Scoring the classifier on the prediction list 1.")
    score_time_start = time.clock()
    linear_score = svm_clf.score(X_test, y_test)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    # Output classification report and confusion matrix.
    name = "linear8({})".format(i+1)
    evmodel.report(y_test, y_pred, n_classes, name)
    evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
    evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)

    # Output the results.
    with open('results/{}_round{}'.format(name, i+1), "w") as text_file:
        print(name, file=text_file)
        print("> {:.2f} percent correct.".format(linear_score*100),
              file=text_file)
        print("> Fit time {}".format(fit_time), file=text_file)
        print("> Test time {}".format(score_time), file=text_file)

    # Append this rounds scores and times to the respective metascore array.
    linear_scores.append(linear_score)
    linear_fit_times.append(fit_time)
    linear_score_times.append(score_time)

    """ LINEAR KERNEL // LIST 2 // LANDMARKS """
    svm_clf = OneVsRestClassifier(lin_svm_clf)

    print("***> Training SVM with LINEAR kernel & list 2.")
    fit_time_start = time.clock()
    y_pred = svm_clf.fit(X_train1, y_train1).predict(X_test1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    linear_score = svm_clf.score(X_test1, y_test1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "linear5({})".format(i+1)
    evmodel.report(y_test1, y_pred, n_classes1, name)
    evmodel.matrix(y_test1, y_pred, np.unique(y_train1), False, name)
    evmodel.matrix(y_test1, y_pred, np.unique(y_train1), True, name)

    with open('results/{}_round{}'.format(name, i+1), "w") as text_file:
        print(name, file=text_file)
        print("> {:.2f} percent correct.".format(linear_score*100),
              file=text_file)
        print("> Fit time {}".format(fit_time), file=text_file)
        print("> Test time {}".format(score_time), file=text_file)

    linear_scores1.append(linear_score)
    linear_fit_times1.append(fit_time)
    linear_score_times1.append(score_time)

    """ RBF KERNEL // LIST 1 // LANDMARKS """
    rbf_clf = OneVsRestClassifier(rbf_svm_clf)

    print("***> Training SVM with RBF kernel & list 1.")
    fit_time_start = time.clock()
    y_pred = rbf_clf.fit(X_train, y_train).predict(X_test)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 1.")
    score_time_start = time.clock()
    rbf_score = rbf_clf.score(X_test, y_test)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "rbf8({})".format(i+1)
    evmodel.report(y_test, y_pred, n_classes, name)
    evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
    evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)

    with open('results/{}_round{}'.format(name, i+1), "w") as text_file:
        print(name, file=text_file)
        print("> {:.2f} percent correct.".format(rbf_score*100),
              file=text_file)
        print("> Fit time {}".format(fit_time), file=text_file)
        print("> Test time {}".format(score_time), file=text_file)

    rbf_scores.append(rbf_score)
    rbf_fit_times.append(fit_time)
    rbf_score_times.append(score_time)

    """ RBF KERNEL // LIST 2 // LANDMARKS """
    rbf_clf = OneVsRestClassifier(rbf_svm_clf)

    print("Training SVM with RBF kernel & list 2.")
    fit_time_start = time.clock()
    y_pred = rbf_clf.fit(X_train1, y_train1).predict(X_test1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    rbf_score = rbf_clf.score(X_test1, y_test1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "rbf5({})".format(i+1)
    evmodel.report(y_test1, y_pred, n_classes1, name)
    evmodel.matrix(y_test1, y_pred, np.unique(y_train1), False, name)
    evmodel.matrix(y_test1, y_pred, np.unique(y_train1), True, name)

    with open('results/{}_round{}'.format(name, i+1), "w") as text_file:
        print(name, file=text_file)
        print("> {:.2f} percent correct.".format(rbf_score*100),
              file=text_file)
        print("> Fit time {}".format(fit_time), file=text_file)
        print("> Test time {}".format(score_time), file=text_file)

    rbf_scores1.append(rbf_score)
    rbf_fit_times1.append(fit_time)
    rbf_score_times1.append(score_time)

    """ POLY KERNEL // LIST 1 // LANDMARKS """
    poly_clf = OneVsRestClassifier(poly_svm_clf)

    print("Training SVM with POLYNOMIAL kernel & list 1.")
    fit_time_start = time.clock()
    y_pred = poly_clf.fit(X_train, y_train).predict(X_test)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 1.")
    score_time_start = time.clock()
    poly_score = poly_clf.score(X_test, y_test)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "poly8({})".format(i+1)
    evmodel.report(y_test, y_pred, n_classes, name)
    evmodel.matrix(y_test, y_pred, np.unique(y_train), False, name)
    evmodel.matrix(y_test, y_pred, np.unique(y_train), True, name)

    with open('results/{}_round{}'.format(name, i+1), "w") as text_file:
        print(name, file=text_file)
        print("> {:.2f} percent correct.".format(poly_score*100),
              file=text_file)
        print("> Fit time {}".format(fit_time), file=text_file)
        print("> Test time {}".format(score_time), file=text_file)

    poly_scores.append(poly_score)
    poly_fit_times.append(fit_time)
    poly_score_times.append(score_time)

    """ POLY KERNEL // LIST 2 """
    poly_clf = OneVsRestClassifier(poly_svm_clf)

    print("Training SVM with POLYNOMIAL kernel & list 2.")
    fit_time_start = time.clock()
    y_pred = poly_clf.fit(X_train1, y_train1).predict(X_test1)
    fit_time_stop = time.clock()
    fit_time = fit_time_stop-fit_time_start

    print("Scoring the classfier on the prediction set 2.")
    score_time_start = time.clock()
    poly_score = poly_clf.score(X_test1, y_test1)
    score_time_stop = time.clock()
    score_time = score_time_stop-score_time_start

    name = "poly5({})".format(i+1)
    evmodel.report(y_test1, y_pred, n_classes1, name)
    evmodel.matrix(y_test1, y_pred, np.unique(y_train1), False, name)
    evmodel.matrix(y_test1, y_pred, np.unique(y_train1), True, name)

    with open('results/{}_round{}'.format(name, i+1), "w") as text_file:
        print(name, file=text_file)
        print("> {:.2f} percent correct.".format(poly_score*100),
              file=text_file)
        print("> Fit time {}".format(fit_time), file=text_file)
        print("> Test time {}".format(score_time), file=text_file)

    poly_scores1.append(poly_score)
    poly_fit_times1.append(fit_time)
    poly_score_times1.append(score_time)

    """ END OF THE ROUND """

    with open('results/output{}'.format(i+1), "w") as text_file:
        print("\n{}: Test round {} - ".format(script_name, (i+1)),
              file=text_file)
        print("\nSVM, LINEAR, Landmarks, List 1:",
              file=text_file)
        print("> {:.2f} percent correct.".format(linear_scores[i]*100),
              file=text_file)
        print("> Fit time {}".format(linear_fit_times[i]),
              file=text_file)
        print("> Test time {}".format(linear_score_times[i]),
              file=text_file)
        print("\nSVM, LINEAR, Landmarks, List 2:",
              file=text_file)
        print("> {:.2f} percent correct.".format(linear_scores1[i]*100),
              file=text_file)
        print("> Fit time {}".format(linear_fit_times1[i]),
              file=text_file)
        print("> Test time {}".format(linear_score_times1[i]),
              file=text_file)
        print("\nSVM, RBF, Landmarks, List 1:",
              file=text_file)
        print("> {:.2f} percent correct.".format(rbf_scores[i]*100),
              file=text_file)
        print("> Fit time {}".format(rbf_fit_times[i]),
              file=text_file)
        print("> Test time {}".format(rbf_score_times[i]),
              file=text_file)
        print("\nSVM, RBF, Landmarks, List 2:",
              file=text_file)
        print("> {:.2f} percent correct.".format(rbf_scores1[i]*100),
              file=text_file)
        print("> Fit time {}".format(rbf_fit_times1[i]),
              file=text_file)
        print("> Test time {}".format(rbf_score_times1[i]),
              file=text_file)
        print("\nSVM, POLYNOMIAL, Landmarks, List 1:",
              file=text_file)
        print("> {:.2f} percent correct.".format(poly_scores[i]*100),
              file=text_file)
        print("> Fit time {}".format(poly_fit_times[i]),
              file=text_file)
        print("> Test time {}".format(poly_score_times[i]),
              file=text_file)
        print("\nSVM, POLYNOMIAL, Landmarks, List 2:",
              file=text_file)
        print("> {:.2f} percent correct.".format(poly_scores1[i]*100),
              file=text_file)
        print("> Fit time {}".format(poly_fit_times1[i]),
              file=text_file)
        print("> Test time {}".format(poly_score_times1[i]),
              file=text_file)

""" END OF THE TEST """

with open('results/output', "w") as text_file:
    print("\nAverages for SVM with Linear kernel / set 1 - {:.2f}%, {}, {}."
          .format((np.mean(linear_scores)*100),
                  (np.mean(linear_fit_times)),
                  (np.mean(linear_score_times))), file=text_file)
    print("Averages for SVM with Linear kernel / set 2 - {:.2f}%, {}, {}."
          .format((np.mean(linear_scores1)*100),
                  (np.mean(linear_fit_times1)),
                  (np.mean(linear_score_times1))), file=text_file)
    print("\nAverages for SVM with RBF kernel / set 1 - {:.2f}%, {}, {}."
          .format((np.mean(rbf_scores)*100),
                  (np.mean(rbf_fit_times)),
                  (np.mean(rbf_score_times))), file=text_file)
    print("Averages for SVM with RBF kernel / set 2 - {:.2f}%, {}, {}."
          .format((np.mean(rbf_scores1)*100),
                  (np.mean(rbf_fit_times1)),
                  (np.mean(rbf_score_times1))), file=text_file)
    print("\nAverages for SVM with POLY kernel / set 1 - {:.2f}%, {}, {}."
          .format((np.mean(poly_scores)*100),
                  (np.mean(poly_fit_times)),
                  (np.mean(poly_score_times))), file=text_file)
    print("Averages for SVM with POLY kernel / set 2 - {:.2f}%, {}, {}."
          .format((np.mean(poly_scores1)*100),
                  (np.mean(poly_fit_times1)),
                  (np.mean(poly_score_times1))), file=text_file)

# End the script.
end = time.clock()
print("""\n{}: Program end. Time elapsed:
      {:.5f}.""".format(script_name, end - start))

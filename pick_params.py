#!/usr/bin/env python3

"""Select best possible values of C, Gamma, Degree, Coef0 for SVMs."""

# Import packages.
import os
import time
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# Scikit imports.
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV

# My imports.
import extraction_model as exmodel
from sort_database.utils import EMOTIONS_8, MidpointNormalize


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: Picking best parameters for classifiers...\n".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate.


# Parameters to test.
Cs = [0.01, 0.1, 1, 5, 10]
gammas = [0.01, 0.1, 1, 5, 10]
degreeValues = [1, 2, 3]
coef0Values = [0, 1, 2]


def linear_param_selection(X, y):
    """Find best parameters for linear kernel."""
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(LinearSVC(loss="hinge"), param_grid)
    grid_search.fit(X, y)
    return grid_search


def rbf_param_selection(X, y):
    """Find best parameters for rbf kernel."""
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(X, y)
    return grid_search


def poly_param_selection(X, y):
    """Find best parameters for polynomial kernel."""
    param_grid = {'C': Cs, 'gamma': gammas,
                  'degree': degreeValues, 'coef0': coef0Values}
    grid_search = GridSearchCV(SVC(kernel='poly'), param_grid)
    grid_search.fit(X, y)
    return grid_search


# Create Grid for each kernel and create the dataset.
X_train, y_train, X_test, y_test = exmodel.get_sets(EMOTIONS_8)
X = X_train + X_test
y = y_train + y_test
lin_dict = rbf_dict = poly_dict = []


def thread1():
    """Run the linear parameter selection."""
    global lin_dict
    lin_dict = linear_param_selection(X, y)


def thread2():
    """Run the rbf parameter selection."""
    global rbf_dict
    rbf_dict = rbf_param_selection(X, y)


def thread3():
    """Run the polynomial parameter selection."""
    global poly_dict
    poly_dict = poly_param_selection(X, y)


# Run tasks using processes.
processes = [multiprocessing.Process(target=thread1()),
             multiprocessing.Process(target=thread2()),
             multiprocessing.Process(target=thread3())]
[process.start() for process in processes]
[process.join() for process in processes]

# Print selections to file.
w = csv.writer(open('results/lin_params.csv', "w"))
for key, val in lin_dict.best_params_.items():
    w.writerow([key, val])
w = csv.writer(open('results/rbf_params.csv', "w"))
for key, val in rbf_dict.best_params_.items():
    w.writerow([key, val])
w = csv.writer(open('results/poly_params.csv', "w"))
for key, val in poly_dict.best_params_.items():
    w.writerow([key, val])

# Print selections to terminal.
print("\nThe best linear parameters are: ", lin_dict.best_params_)
print("The best radial basis function parameters are: ", rbf_dict.best_params_)
print("The best polynomial parameters are: ", poly_dict.best_params_)

# Extract the scores of rbf kernal.
rbf_scores = rbf_dict.cv_results_['mean_test_score'].reshape(len(Cs),
                                                             len(gammas))

# Make a nice figure to show rbf kernel's different parameter results.
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(rbf_scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gammas)), gammas, rotation=45)
plt.yticks(np.arange(len(Cs)), Cs)
plt.title('Validation accuracy')
plt.savefig('results/rbf_params')

# End the script.
end = time.clock()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("\n***> Time elapsed: {:0>2}:{:0>2}:{:05.2f}."
      .format(int(hours), int(minutes), seconds))

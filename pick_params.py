#!/usr/bin/env python3

"""Select best possible values of C, Gamma, Degree, Coef0 for SVMs."""

# Import packages.
import os
import time
import numpy as np
import pylab as pl
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV

# My imports.
import extraction_model as exmodel
from sort_database.utils import EMOTIONS_8


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: ...\n".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate.


# Parameters.
Cs = [0.01, 0.1, 1, 5, 10]
gammas = [0.01, 0.1, 1, 5, 10]


def rbf_param_selection(X, y):
    """Find best parameters for rbf kernel."""
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid)
    grid_search.fit(X, y)
    return grid_search


def linear_param_selection(X, y):
    """Find best parameters for linear kernel."""
    param_grid = {'C': Cs}
    grid_search = GridSearchCV(LinearSVC(loss="hinge"), param_grid)
    grid_search.fit(X, y)
    return grid_search


def poly_param_selection(X, y):
    """Find best parameters for polynomial kernel."""
    degreeValues = [1, 2, 3]
    coef0Values = [0, 1, 2]
    param_grid = {'C': Cs, 'gamma': gammas,
                  'degree': degreeValues, 'coef0': coef0Values}
    grid_search = GridSearchCV(SVC(kernel='poly'), param_grid)
    grid_search.fit(X, y)
    return grid_search


# Create Grid for each kernel.
X_train, y_train, X_test, y_test = exmodel.get_sets(EMOTIONS_8)
X = X_train + X_test
y = y_train + y_test
lin_dict = linear_param_selection(X, y)
rbf_dict = rbf_param_selection(X, y)
poly_dict = poly_param_selection(X, y)

# Print selections to file.
with open('results/svm_params', "w") as text_file:
    print(lin_dict.best_params_)
    print(rbf_dict.best_params_)
    print(poly_dict.best_params_)

# Extract just the scores.
lin_scores = [x[1] for x in lin_dict]
rbf_scores = [x[1] for x in rbf_dict]
poly_scores = [x[1] for x in poly_dict]

# Reshape the scores.
lin_scores = np.array(lin_scores).reshape(len(Cs), len(gammas))
rbf_scores = np.array(rbf_scores).reshape(len(Cs), len(gammas))
poly_scores = np.array(poly_scores).reshape(len(Cs), len(gammas))

# Make a nice figure to show.
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(rbf_scores, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gammas)), gammas, rotation=45)
pl.yticks(np.arange(len(Cs)), Cs)
pl.savefig('results/rbf_params')

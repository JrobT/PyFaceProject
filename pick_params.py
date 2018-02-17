"""Selection best possible values of C and Gamma for rbf kernel function."""

# Import packages.
import numpy as np
import pylab as pl
import os
import time

# Scikit.
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# My imports.
import extraction_model as exmodel
from sort_database.utils import EMOTIONS_5


print(__doc__)

# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\n{}: Beginning Support Vector Machine tests...\n".format(script_name))
start = time.clock()  # Start of the speed test. clock() is most accurate

X_train, y_train, X_test, y_test = exmodel.get_sets(EMOTIONS_5)


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


# Plot the scores of the grid.
# grid_scores_ contains parameter settings and scores.
score_dict = grid.grid_scores_

# We extract just the scores.
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))

# Make a nice figure.
pl.figure(figsize=(8, 6))
pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
pl.xlabel('gamma')
pl.ylabel('C')
pl.colorbar()
pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
pl.yticks(np.arange(len(C_range)), C_range)
pl.show()

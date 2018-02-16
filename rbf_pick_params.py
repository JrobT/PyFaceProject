"""Selection best possible values of C and Gamma for rbf kernel function."""

# Import packages.
import numpy as np
import pylab as pl

# Scikit.
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# My imports.
import extraction_model as exmodel
from constants import EMOTIONS_5


print(__doc__)

X_train, y_train, X_test, y_test = exmodel.get_sets(EMOTIONS_5)

# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = 10. ** np.arange(-3, 8)
gamma_range = 10. ** np.arange(-5, 4)

parameters = {'kernel': ('linear', 'rbf'), 'gamma': gamma_range, 'C': C_range}

grid = GridSearchCV(SVC(), param_grid=parameters)
grid.fit(X_train, y_train)
print("The best classifier is: ", grid.best_estimator_)

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

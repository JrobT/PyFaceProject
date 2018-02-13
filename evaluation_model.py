#!/usr/bin/env python3

"""
Model contains different methods of evaluating the classifiers.

ROC curves aren't the best method for multiclass classification problems. I
will try to implement these ROC curves, however I also provide other methods
which are better for this kind of problem.

Included are: Classification Report, Confusion Matrix
"""

# Import packages.
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Import evaluation methods.
from sklearn.metrics import classification_report, confusion_matrix

# My imports.
from constants import EMOTIONS_5, EMOTIONS_8


def report(y_test, y_pred, n_classes, save_name):
    """Produce and print the classification report to file with save_name."""
    if n_classes == 5:
        target_names = EMOTIONS_5
    else:
        target_names = EMOTIONS_8
    with open('results/classreport_{}'.format(save_name), "w") as text_file:
        print(classification_report(y_test, y_pred, target_names=target_names),
              file=text_file)


def matrix(y_test, y_pred, classes, normalize, save_name, cmap=plt.cm.Blues):
    """Produce and print the confusion matrix to file with save_name."""
    # Compute confusion matrix.
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.matshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(save_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if normalize:
        norm = "_normalized"
    else:
        norm = ""

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion{}_{}'.format(norm, save_name))


# def build_roc_curve(n_classes, y_score, y_test, title):
#     """Build a ROC curve for a multiclass problem."""
#     # Compute ROC curve and ROC area for each class.
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(0, n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area.
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # Plot ROC curve.
#     fig = plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#              ''.format(roc_auc["micro"]))
#     for i in range(n_classes):
#         plt.plot(fpr[i], tpr[i],
#                  label='ROC curve of class {0} (area = {1:0.2f})'
#                        ''.format(i+1, roc_auc[i]))
#
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Multiclass Receiver Operating Characteristic Curve for {}'
#               ''.format(title))
#     plt.legend(loc="lower right")
#     fig.savefig('roc_curves_reports/{}'.format(title))

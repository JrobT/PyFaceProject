#!/usr/bin/env python3

"""Feature extraction via pca and lda model."""

# Import packages.
import os
import sys
import numpy as np

from PIL import Image


def asRowMatrix(X):
    """Return data by row."""
    if (len(X) == 0):
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def asColumnMatrix(X):
    """Return data by column."""
    if (len(X) == 0):
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


def read_images(path, size=None):
    """Return arrays of images read from files."""
    c = 0
    X, y = [], []
    for (dirname, dirnames, filenames) in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error ({0}): {1}".format(errno, strerror))
                except:
                    print("Unexpected error: ", sys.exc_info()[0])
                    raise
            c = c+1
            return [X, y]


def project(W, X, mu=None):
    """Rearranged projection equation."""
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    """Rearranged reconstruct equation."""
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


def pca(X, y, num_components=0):
    """Principal Component Analysis."""
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu

    if (n > d):
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

    # Sort eigenvectors descending by their eigenvalue.
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select only num_components.
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:, 0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]


path = "sort_database//database"
imgs = read_images(path)

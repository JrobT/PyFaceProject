#!/usr/bin/env python3

"""Variables that are the same and frequently used.

All paths here must be relative from the main project directory.
"""

# Import packages.
import numpy as np
import cv2
from PIL import Image
from matplotlib.colors import Normalize

# Emotion Lists.
EMOTIONS_8 = ["neutral", "anger", "contempt", "disgust",
              "fear", "happy", "sadness", "surprise"]
EMOTIONS_5 = ["anger", "contempt", "disgust", "happy", "surprise"]

# HAAR Cascade Face Classifier Locations. HAAR2, HAAR3, HAAR4 not needed.
HAAR = "OpenCV_HAAR_CASCADES//haarcascade_frontalface_default.xml"
HAAR2 = "OpenCV_HAAR_CASCADES//haarcascade_frontalface_alt2.xml"
HAAR3 = "OpenCV_HAAR_CASCADES//haarcascade_frontalface_alt.xml"
HAAR4 = "OpenCV_HAAR_CASCADES//haarcascade_frontalface_alt_tree.xml"

# dlib's Shape Predictor Location.
PRED = "FaceLandmarks.dat"

dim = (380, 380)


def resize(gray):
    """Return the given image, resized to 380x380."""
    out = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
    # cv2.imwrite('resized.png', out)
    return out


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = Image.open(open(pic, 'rb')).convert('LA')  # Grayscale
    basewidth = 380  # Resize to 380 width
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(pic, "PNG")  # Save as .png


def matrix_image(image):
    """Open image and convert it to a resized matrix."""
    image = Image.open(image)
    image = image.resize(dim)
    image = list(image.getdata())
    image = map(list, image)
    image = np.array(image)
    return image


class MidpointNormalize(Normalize):
    """Move the midpoint of colour map around values of interest."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """Initialise utility function."""
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """If called, do this."""
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#!/usr/bin/env python3

"""Variables that are the same and frequently used.

All paths here must be relative from the main project directory.
"""

# Import packages.
from PIL import Image

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


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = Image.open(open(pic, 'rb')).convert('LA')  # Grayscale
    basewidth = 380  # Resize to 380 width
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(pic, "PNG")  # Save as .png

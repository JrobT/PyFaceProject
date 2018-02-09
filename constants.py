#!/usr/bin/env python3

"""Variables that are the same and frequently used."""

# Emotion Lists.
EMOTIONS_8 = ["neutral", "anger", "contempt", "disgust",
              "fear", "happy", "sadness", "surprise"]
EMOTIONS_5 = ["anger", "contempt", "disgust", "happy", "surprise"]

# HAAR Cascade Face Classifier Locations. HAAR2, HAAR3, HAAR4 not needed.
HAAR = "sort_database//OpenCV_HAAR_CASCADES//haarcascade_frontalface_default.xml"
HAAR2 = "sort_database//OpenCV_HAAR_CASCADES/haarcascade_frontalface_alt2.xml"
HAAR3 = "sort_database//OpenCV_HAAR_CASCADES/haarcascade_frontalface_alt.xml"
HAAR4 = "sort_database//OpenCV_HAAR_CASCADES/haarcascade_frontalface_alt_tree.xml"

# dlib's Shape Predictor Location.
PRED = "sort_database//FaceLandmarks.dat"

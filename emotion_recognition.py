#!/usr/bin/env python3

"""."""

import numpy as np  # SVC methods will accept numpy arrays
import cv2
import dlib
import glob
import math

# Importing Scikit-Learn.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib

from constants import HAAR, HAAR2, HAAR3, HAAR4


class SVM:
    """My Support Vector Machine."""

    def __init__(self):
        """Initialise required variables."""
        self.emotions = ["anger", "contempt", "disgust", "happy", "surprise"]
        # Build the required objects.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Set Face Detectors.
        self.faceDet = cv2.CascadeClassifier(HAAR)
        self.faceDet2 = cv2.CascadeClassifier(HAAR2)
        self.faceDet3 = cv2.CascadeClassifier(HAAR3)
        self.faceDet4 = cv2.CascadeClassifier(HAAR4)
        self.faceDet5 = dlib.get_frontal_face_detector()  # dlib's face detector
        self.predictor = dlib.shape_predictor("sort_database//FaceLandmarks.dat")  # file must be in dir
        self.landmark_data = {}  # Make dictionary for all data point values
        # Build classifier model.
        self.model = Pipeline((
            ("scaler", StandardScaler()),
            ("svc_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
        ))

    def get_images(self, emotion):
        """Get all the images to train on."""
        files = glob.glob("sort_database//database//{}//*".format(emotion))
        print("Training {} images with labelled with {}".format(len(files), emotion))
        return files

    def get_landmarks(self, image):
        """Overlay the predictor and turn into features for the classifier."""
        # Use detectors to detect faces in the frame.
        detections = self.faceDet5(image, 1)

        haar_detections = []
        if not len(detections) > 0:  # dlib's detector will work over 50% of the time
            haar_detections = self.faceDet.detectMultiScale(image, scaleFactor=1.1,
                                                            minNeighbors=10, minSize=(5, 5),
                                                            flags=cv2.CASCADE_SCALE_IMAGE)
            haar_detections2 = self.faceDet2.detectMultiScale(image, scaleFactor=1.1,
                                                              minNeighbors=10, minSize=(5, 5),
                                                              flags=cv2.CASCADE_SCALE_IMAGE)
            haar_detections3 = self.faceDet3.detectMultiScale(image, scaleFactor=1.1,
                                                              minNeighbors=10, minSize=(5, 5),
                                                              flags=cv2.CASCADE_SCALE_IMAGE)
            haar_detections4 = self.faceDet4.detectMultiScale(image, scaleFactor=1.1,
                                                              minNeighbors=10, minSize=(5, 5),
                                                              flags=cv2.CASCADE_SCALE_IMAGE)

            if len(haar_detections) > 0:
                for (x, y, w, h) in haar_detections:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break  # if found, no point in making another feature vector
            elif len(haar_detections2) > 0:
                for (x, y, w, h) in haar_detections2:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break
            elif len(haar_detections3) > 0:
                for (x, y, w, h) in haar_detections3:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break
            elif len(haar_detections4) > 0:
                for (x, y, w, h) in haar_detections4:
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                    detections.append(dlib_rect)
                    break

        # We may detect 0, 1 or many faces. Loop through each face detected.
        for i, j in enumerate(detections):
            # Draw facial landmarks with the predictor class.
            shape = self.predictor(image, j)
            xlist = []
            ylist = []

            # Store X and Y coordinates in separate lists.
            for i in range(1, 68):  # 68 because we're looking for 68 landmarks
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            # Find both coordinates for the centre of gravity (middle point).
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)

            # Calculate the distance from centre to other points in both axes.
            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]

            # Condition the vectors.
            landmarks_vectorised = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(w)
                landmarks_vectorised.append(z)
                meannp = np.asarray((ymean, xmean))
                coornp = np.asarray((z, w))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorised.append(dist)
                landmarks_vectorised.append((math.atan2(y, x)*360)/(math.pi*2))

            self.landmark_data['landmarks_vectorised'] = landmarks_vectorised
        if len(detections) < 1:
            self.landmark_data['landmarks_vectorised'] = "error"

    def sets(self, emotions):
        """Make the sets to train on."""
        training_data = []
        training_labels = []

        for emotion in emotions:
            training = self.get_images(emotion)

            # Append data to training and prediction lists and label 0 to num.
            for item in training:
                image = cv2.imread(item)  # Open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                clahe_image = self.clahe.apply(gray)
                self.get_landmarks(clahe_image)

                if self.landmark_data['landmarks_vectorised'] != "error":
                    training_data.append(self.landmark_data['landmarks_vectorised'])
                    training_labels.append(emotions.index(emotion))

        print("Training Data: {}, Labels: {}".format(len(training_data), len(training_labels)))

        return training_data, training_labels

    def train(self):
        """Train the SVM."""
        tdata, tlabels = self.sets(self.emotions)
        nptdata = np.array(tdata)
        print("Fitting the classifier")
        self.model.fit(nptdata, tlabels)

    def predict(self, image):
        """Use SVM to predict class."""
        if image is None:
            return None
        return self.emotions[int(self.model.predict(image))]

    def save(self):
        """Persist the classifier once its been trained."""
        joblib.dump(self.model, 'svm.pkl')

    def load(self):
        """Load the persisted model."""
        self.model = joblib.load('svm.pkl')

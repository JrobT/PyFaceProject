#!/usr/bin/env python3

"""."""

import cv2

from constants import EMOTIONS_5, HAAR, HAAR2, HAAR3, HAAR4
from emotion_recognition import SVM

# Set Face Detectors.
faceDet = cv2.CascadeClassifier(HAAR)
# faceDet2 = cv2.CascadeClassifier(HAAR2)
# faceDet3 = cv2.CascadeClassifier(HAAR3)
# faceDet4 = cv2.CascadeClassifier(HAAR4)


def format_image(image):
    """Format the video frame to an image the classfier can predict from."""
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    # Look for faces.
    if not faceDet.empty():
        faces = faceDet.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    else:
        faces = []
    # faces.append(faceDet2.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5))
    # faces.append(faceDet.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5))
    # faces.append(faceDet.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5))

    # None means no face found.
    if not len(faces) > 0:
        return None, None

    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    # Chop image to face.
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resize image to classfier size.
    try:
        image = cv2.resize(image, (380, 380), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        return None
    return image


# Load Model
SVM = SVM()
SVM.train()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

emojis = []
for emotion in EMOTIONS_5:
    emojis.append(cv2.imread('emojis//' + emotion + '.png'))

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Predict result with classifier
    result = SVM.predict(format_image(frame))

    # Draw face in frame
    # if faces is not None:
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Write results in frame
    if result is not None:
        for index, emotion in enumerate(EMOTIONS_5):
            cv2.putText(frame, emotion, (10, index * 20 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10),
                          (130 + int(result[0][index] * 100),
                          (index + 1) * 20 + 4), (255, 0, 0), -1)

        # face_image = emojis[result[0].index(max(result[0]))]
        #
        # for c in range(0, 3):
        #     frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

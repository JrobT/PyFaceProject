#!/usr/bin/env python3
"""Python script to train the classifiers and test them on different sets.

I have defined 4 sets:
    -   Full Emotion List (8)
    -   Emotion list without Fear & Surprise, as they have the least images (6)
    -   The Emotion list without Neutral (7)
    -   The Emotion list without Fear, Surprise, and Neutral (5)

I have 3 classifiers:
    -   Fisher Faces
    -   Eigenfaces
    -   Local Binary Pattern Histograms
"""
import cv2  # OpenCV
import glob
import random
import os
import time
import multiprocessing
import numpy as np


# Start the script.
sname = os.path.basename(__file__)  # The name of this script
print("\n\n%s: Program starts..." % sname)
start = time.clock()

el = ["neutral", "anger", "contempt", "disgust",
      "fear", "happy", "sadness", "surprise"]  # The emotion list
el2 = ["neutral", "anger", "contempt", "disgust",
       "happy", "sadness"]  # The list without Fear and Surprise
el3 = ["anger", "contempt", "disgust",
       "fear", "happy", "sadness", "surprise"]  # The list without Neutral
el4 = ["anger", "contempt", "disgust",
       "happy", "sadness"]  # The list without Fear, Surprise and Neutral

fisherface = cv2.face.FisherFaceRecognizer_create()  # Fisherface classifier
eigenface = cv2.face.EigenFaceRecognizer_create()  # Eigenface classifier
lbph = cv2.face.LBPHFaceRecognizer_create()  # Local Binary Patterns classifier

data = {}


def get_images(emotion):
    """Split dataset into 80 percent training set and 20 percent prediction."""
    files = glob.glob("sort_database//database//%s//*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


def sets(emotions):
    """Make the sets to test on."""
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        training, prediction = get_images(emotion)

        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_fisher_recognizer(training_data, training_labels,
                          prediction_data, prediction_labels):
    """Run the training."""
    print("\n%s: Training fisherface classifier..." % sname)
    print("%s: Size of the training set is %s images"
          % (sname, len(training_labels)))

    fisherface.train(training_data, np.asarray(training_labels))

    print("%s: Predicting classification set..." % sname)
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fisherface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / cnt)


def run_eigen_recognizer(training_data, training_labels,
                         prediction_data, prediction_labels):
    """Run the training."""
    print("\n%s: Training eigenface classifier..." % sname)
    print("%s: Size of the training set is %s images"
          % (sname, len(training_labels)))

    eigenface.train(training_data, np.asarray(training_labels))

    print("%s: Predicting classification set..." % sname)
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fisherface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / cnt)


def run_lbph_recognizer(training_data, training_labels,
                        prediction_data, prediction_labels):
    """Run the training."""
    print("\n%s: Training Local Binary Pattern classifier..." % sname)
    print("%s: Size of the training set is %s images"
          % (sname, len(training_labels)))

    lbph.train(training_data, np.asarray(training_labels))

    print("%s: Predicting classification set..." % sname)
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fisherface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / cnt)


training_data, training_labels, prediction_data, prediction_labels = sets(el)
training_data2, training_lbls2, prediction_data2, prediction_lbls2 = sets(el2)
training_data3, training_lbls3, prediction_data3, prediction_lbls3 = sets(el3)
training_data4, training_lbls4, prediction_data4, prediction_lbls4 = sets(el4)

metascore1 = []
metascore2 = []
metascore3 = []
metascore4 = []
metascore5 = []
metascore6 = []
metascore7 = []
metascore8 = []
metascore9 = []
metascore10 = []
metascore11 = []
metascore12 = []


def threadme1():
    """Produce the results."""
    for i in range(0, 5):
        f = run_fisher_recognizer(training_data, training_labels,
                                  prediction_data, prediction_labels)
        print("%s: F got %s percent correct!" % (sname,
                                                 f))
        metascore1.append(f)


def threadme2():
    """Produce the results."""
    for i in range(0, 5):
        f2 = run_fisher_recognizer(training_data2, training_lbls2,
                                   prediction_data2, prediction_lbls2)
        print("%s: F2 got %s percent correct!" % (sname,
                                                  f2))
        metascore2.append(f2)


def threadme3():
    """Produce the results."""
    for i in range(0, 5):
        f3 = run_fisher_recognizer(training_data3, training_lbls3,
                                   prediction_data3, prediction_lbls3)
        print("%s: F3 got %s percent correct!" % (sname,
                                                  f3))
        metascore3.append(f3)


def threadme4():
    """Produce the results."""
    for i in range(0, 5):
        f4 = run_fisher_recognizer(training_data4, training_lbls4,
                                   prediction_data4, prediction_lbls4)
        print("%s: F4 got %s percent correct!" % (sname,
                                                  f4))
        metascore4.append(f4)


def threadme5():
    """Produce the results."""
    for i in range(0, 5):
        e = run_eigen_recognizer(training_data, training_labels,
                                 prediction_data, prediction_labels)
        print("%s: E got %s percent correct!" % (sname,
                                                 e))
        metascore5.append(e)


def threadme6():
    """Produce the results."""
    for i in range(0, 5):
        e2 = run_eigen_recognizer(training_data2, training_lbls2,
                                  prediction_data2, prediction_lbls2)
        print("%s: E2 got %s percent correct!" % (sname,
                                                  e2))
        metascore6.append(e2)


def threadme7():
    """Produce the results."""
    for i in range(0, 5):
        e3 = run_eigen_recognizer(training_data3, training_lbls3,
                                  prediction_data3, prediction_lbls3)
        print("%s: E3 got %s percent correct!" % (sname,
                                                  e3))
        metascore7.append(e3)


def threadme8():
    """Produce the results."""
    for i in range(0, 5):
        e4 = run_eigen_recognizer(training_data4, training_lbls4,
                                  prediction_data4, prediction_lbls4)
        print("%s: E4 got %s percent correct!" % (sname,
                                                  e4))
        metascore8.append(e4)


def threadme9():
    """Produce the results."""
    for i in range(0, 5):
        l = run_lbph_recognizer(training_data, training_labels,
                                prediction_data, prediction_labels)
        print("%s: L got %s percent correct!" % (sname,
                                                 l))
        metascore9.append(l)


def threadme10():
    """Produce the results."""
    for i in range(0, 5):
        l2 = run_lbph_recognizer(training_data2, training_lbls2,
                                 prediction_data2, prediction_lbls2)
        print("%s: L2 got %s percent correct!" % (sname,
                                                  l2))
        metascore10.append(l2)


def threadme11():
    """Produce the results."""
    for i in range(0, 5):
        l3 = run_lbph_recognizer(training_data3, training_lbls3,
                                 prediction_data3, prediction_lbls3)
        print("%s: L3 got %s percent correct!" % (sname,
                                                  l3))
        metascore11.append(l3)


def threadme12():
    """Produce the results."""
    for i in range(0, 5):
        l4 = run_lbph_recognizer(training_data4, training_lbls4,
                                 prediction_data4, prediction_lbls4)
        print("%s: L4 got %s percent correct!" % (sname,
                                                  l4))
        metascore12.append(l4)


# Run tasks using processes
processes = [multiprocessing.Process(target=threadme1()),
             multiprocessing.Process(target=threadme2()),
             multiprocessing.Process(target=threadme3()),
             multiprocessing.Process(target=threadme4()),
             multiprocessing.Process(target=threadme5()),
             multiprocessing.Process(target=threadme6()),
             multiprocessing.Process(target=threadme7()),
             multiprocessing.Process(target=threadme8()),
             multiprocessing.Process(target=threadme9()),
             multiprocessing.Process(target=threadme10()),
             multiprocessing.Process(target=threadme11()),
             multiprocessing.Process(target=threadme12())]
[process.start() for process in processes]
[process.join() for process in processes]

print("\n\n%s: Final score f: Got %s percent correct!"
      % (sname, np.mean(metascore1)))
print("%s: Final score f2: Got %s percent correct!"
      % (sname, np.mean(metascore2)))
print("%s: Final score f3: Got %s percent correct!"
      % (sname, np.mean(metascore3)))
print("%s: Final score f4: Got %s percent correct!"
      % (sname, np.mean(metascore4)))

print("\n%s: Final score e: Got %s percent correct!"
      % (sname, np.mean(metascore5)))
print("%s: Final score e2: Got %s percent correct!"
      % (sname, np.mean(metascore6)))
print("%s: Final score e3: Got %s percent correct!"
      % (sname, np.mean(metascore7)))
print("%s: Final score e4: Got %s percent correct!"
      % (sname, np.mean(metascore8)))

print("\n%s: Final score l: Got %s percent correct!"
      % (sname, np.mean(metascore9)))
print("%s: Final score l2: Got %s percent correct!"
      % (sname, np.mean(metascore10)))
print("%s: Final score l3: Got %s percent correct!"
      % (sname, np.mean(metascore11)))
print("%s: Final score l4: Got %s percent correct!"
      % (sname, np.mean(metascore12)))

# End the script.
end = time.clock()
print("%s: Program end. Time elapsed: %s." % (sname, end - start))

#!/usr/bin/env python3
"""Python script to combine CK, MMI, JAFFE, and Google Images datasets.

I have taken the CK dataset and split it into 'Emotion' and
'Images' folders within 'CK_Dataset' positioned in my working directory.
Permission via link: http://www.consortium.ri.cmu.edu/ckagree/

The MMI dataset images have been analysed using Amazon Mechanical Turk, and
sorted into folders within 'MMI_Dataset' using another script, included in
the 'MMI_Dataset' folder.
Permission via link: https://mmifacedb.eu/

Each of the Google Images folders are in a similar folder structure inside
'Google_Dataset'.

The JAFFE set has its own sort set out as a .txt, which I parsed using another
script.

Next I convert all pictures to the same extension (PNG), matrix and grayscale,
and copied them to their respective folder in the sorted set.
"""
import cv2  # OpenCV
import csv
import glob
import PIL
import os
import errno
from PIL import Image
from shutil import copyfile


sname = os.path.basename(__file__)  # The name of this script


# Define ordered emotion list.
emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]

# HAAR Cascade Face Classifiers.
haar = "OpenCV_HAAR_CASCADES//_haarcascade_frontalface_default.xml"
haar2 = "OpenCV_HAAR_CASCADES//_haarcascade__frontalface_alt2.xml"
haar3 = "OpenCV_HAAR_CASCADES//_haarcascade__frontalface_alt.xml"
haar4 = "OpenCV_HAAR_CASCADES//_haarcascade__frontalface_alt_tree.xml"

# Set Face Detectors.
faceDet = cv2.CascadeClassifier(haar)
faceDet2 = cv2.CascadeClassifier(haar2)
faceDet3 = cv2.CascadeClassifier(haar3)
faceDet4 = cv2.CascadeClassifier(haar4)


def silent_remove(fname):
    """Silently remove a file that may or may not exist."""
    try:
        os.remove(fname)
    except OSError as e:
        if e.errno != errno.ENOENT:  # 'No such file or directory'
            raise  # re-raise exception if a different error occurred


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = Image.open(pic).convert('LA')  # Grayscale

    basewidth = 490  # Resize
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    img.save(pic, "PNG")  # Save as .png


"""  Cohn-Kanade AU-Coded Expression Database

As taken from : http://www.pitt.edu/~emotion/ck-spread.htm
'The Cohn-Kanade AU-Coded Facial Expression Database is for research in
automatic facial image analysis and synthesis and for perceptual studies.
Cohn-Kanade is available in two versions and a third is in preparation.'

I am using the first version. Version 1, the initial release,
includes 486 sequences from 97 posers. Each sequence begins with a neutral
expression and proceeds to a peak expression. The peak expression for each
sequence in fully FACS (Ekman, Friesen, & Hager, 2002; Ekman & Friesen, 1979)
coded and given an emotion label. The emotion label refers to what expression
was requested rather than what may actually have been performed.
For a full description of CK, see (Kanade, Cohn, & Tian, 2000).

I use this dataset with full permission granted.
"""

print("/n/n%s: Processing 'Cohn-Kanade AU-Coded Expression Database'...")

# Return a list of all folders with each participant's number as its file name.
participants = glob.glob('CK_Dataset//Emotion//*')

for person_no in participants:
    person = "%s" % person_no[-4:]  # Store current participant's folder name
    for session in glob.glob("%s//*" % person_no):
        for files in glob.glob("%s//*" % session):
            current = files[20:-30]
            file = open(files, 'r')

            # Emotion coded as a float, so read as float, change to int.
            emotion = int(float(file.readline()))

            # Get path for last image in sequence, which contains the emotion.
            emotion_pic = glob.glob("CK_Dataset//Images//%s//%s//*"
                                    % (person, current))[-1]  # -1 'last'
            neutral_pic = glob.glob("CK_Dataset//Images//%s//%s//*"
                                    % (person, current))[0]  # 0 'first'

            dest_neut = "Combined_Dataset//neutral//%s" % neutral_pic[25:]
            dest_emot = "Combined_Dataset//%s//%s" % (emotions[emotion],
                                                      emotion_pic[25:])

            # Remove the files if they are already in the combined dataset.
            silent_remove(dest_neut)
            silent_remove(dest_emot)

            # All images are kept to a standardised format.
            standardise_image(neutral_pic)
            standardise_image(emotion_pic)

            # Finally, copy the files to the combined dataset.
            copyfile(neutral_pic, dest_neut)
            copyfile(emotion_pic, dest_emot)

            print("%s: Copied %s into %s." % (sname, neutral_pic, dest_neut))
            print("%s: Copied %s into %s." % (sname, emotion_pic, dest_emot))


""" MMI Facial Expression Database

Via https://mmifacedb.eu/:
'The database consists of over 2900 videos and high-resolution still images of
75 subjects. It is fully annotated for the presence of AUs in videos
(event coding), and partially coded on frame-level, indicating for each frame
whether an AU is in either the neutral, onset, apex or offset phase. A small
part was annotated for audio-visual laughters. The database is freely available
to the scientific community.'

I found the Action Units were fairly inconsistent. Therefore I used Amazon's
worker service to classify the images via a questionaire.

I requested an account and was granted permission for use.
"""

print("/n/n%s: Processing 'MMI Facial Expression Database'...")

avg = 0
cnt = 0
temp = 0
imgName = ""

with open('MMI_Dataset//batch.csv') as csvfile:  # Coded using Mechanical Turk
    reader = csv.DictReader(csvfile)
    for row in reader:
        cnt = cnt + 1
        temp = int(row['AvgRate'])
        avg = temp + avg
        imgName = row['Input.image_url']
        fpath = "MMI_Dataset//MMI_Images//%s" % imgName

        if row['Anger'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("anger", imgName)
        elif row['Contempt'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("contempt", imgName)
        elif row['Disgust'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("disgust", imgName)
        elif row['Fear'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("fear", imgName)
        elif row['Happy'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("happy", imgName)
        elif row['Sadness'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("sadness", imgName)
        elif row['Surprise'] == "1":
            dest_emot = "Combined_Dataset//%s//%s" % ("surprise", imgName)

        # Remove the files if they are already in the combined dataset.
        silent_remove(dest_emot)

        # All images are kept to a standardised format.
        standardise_image(fpath)

        # Finally, copy the files to the combined dataset.
        copyfile(fpath, dest_emot)

        print("%s: Read 'batch.csv'. "
              + "Copied %s into %s." % (sname, imgName, dest_emot))

print("%s: Average Ranking of Amazon Mechanical Turk accurancy"
      + "is %s percent." % (sname, (avg / cnt)))


""" JAFFE Database

Via http://www.kasrl.org/jaffe_info.html:
'Contains over 60 Japanese female subjects.'

I agreed to site the database in my report.
"""

print("/n/n%s: Processing 'JAFFE Database'...")

# Convert .txt into .csv. May need to change the first line of .txt file.
with open('JAFFE_Dataset//JAFFE.txt') as fin, open('JAFFE_Dataset//' +
                                                   'JAFFE.csv', 'w') as fout:
    o = csv.writer(fout)
    for line in fin:
        o.writerow(line.split())

# Move JAFFE images into respective sets.
with open('JAFFE_Dataset//JAFFE.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        lst = [row['HAP'], row['SAD'], row['SUR'],
               row['ANG'], row['DIS'], row['FEA']]
        maxx = max(lst)
        index = lst.index(maxx)  # Get the emotion most picked

        if (index >= 3):  # less than 3 means the emotion is 'undecided' IMHO

            imgName = row['PIC']
            fpath = 'JAFFE_Dataset//Images//%s' % imgName

            if (os.path.exists(fpath)):
                if index == 0:
                    dest_emot = 'Combined_Dataset//%s//%s' % ("happy", imgName)
                elif index == 1:
                    dest_emot = 'Combined_Dataset//%s//%s' % ("sadness",
                                                              imgName)
                elif index == 2:
                    dest_emot = 'Combined_Dataset//%s//%s' % ("surprise",
                                                              imgName)
                elif index == 3:
                    dest_emot = 'Combined_Dataset//%s//%s' % ("anger",
                                                              imgName)
                elif index == 4:
                    dest_emot = 'Combined_Dataset//%s//%s' % ("disgust",
                                                              imgName)
                elif index == 5:
                    dest_emot = 'Combined_Dataset//%s//%s' % ("fear",
                                                              imgName)

                dest_emot = dest_emot.split(".")[0]
                dest_emot += ".png"

                # Remove the files if they are already in the combined dataset.
                silent_remove(dest_emot)

                # All images are kept to a standardised format.
                standardise_image(fpath)

                # Finally, copy the files to the combined dataset.
                copyfile(fpath, dest_emot)

                print("%s: Read 'JAFFE.csv'. "
                      + "Copied %s into %s." % (sname, imgName, dest_emot))
            else:
                print("%s: Read 'JAFFE.csv'. File doesn't exist." % sname)
        else:
            print("%s: Read 'JAFFE.csv'. The emotion is unclear." % sname)


"""  Google Images Dataset

I filled out my database by using the Google images search engine. By typing
each emotion into the search, I used the ZIG Lite Chrome extension
(chrome-extension://bedbigoemkinkepgmcmgnapjcahnedmn/out.html?get) to scrape
the page for suitable images.
"""

print("/n/n%s: Processing 'Google Images Dataset'...")

num = 0

for emotion in emotions:
    for f in glob.glob("Google_Dataset//%s//*" % emotion):
        num = num + 1
        dest = "Combined_Dataset//%s//%s.png" % (emotion, num)

        # Remove the files if they are already in the combined dataset.
        silent_remove(dest)

        # All images are kept to a standardised format.
        standardise_image(f)

        # Finally, copy the files to the combined dataset.
        copyfile(f, dest)

        print("%s: Copied %s into %s." % (sname, f, dest))

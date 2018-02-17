#!/usr/bin/env python3

"""Python script to sort the Cohn-Kanade+ AU-Coded Expression Database.

I have taken the CK+ dataset and split it into 'Emotion' and
'Images' folders within 'CK_Dataset' positioned in my working directory.
Permission via link: http://www.consortium.ri.cmu.edu/ckagree/

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
For a description of the extension to CK called CK+, see the website.

I use this dataset with full permission granted. Pictures ©Jeffrey Cohn.
"""

# Import packages.
import glob
from shutil import copyfile

# My imports.
from utils import EMOTIONS_8, standardise_image


dataset = "CKplus"


print("***> Processing Cohn-Kanade+ AU-Coded Expression Database...")

# Return a list of all folders with each participant's number as its file name.
participants = glob.glob("CKplus_dataset//Emotion//*")

filenumber = 0
for person_no in participants:
    person = "%s" % person_no[-4:]  # Store current participant's folder name
    for session in glob.glob("{}//*".format(person_no)):
        for files in glob.glob("{}//*".format(session)):
            current = files[29:-30]
            file = open(files, 'r')

            # Emotion coded as a float, so read as float, change to int.
            emotion = int(float(file.readline()))

            # Get path for last image in sequence, which contains the emotion.
            emotion_pic = glob.glob("CKplus_dataset//Image//{}//{}//*"
                                    .format(person, current))[0]
            neutral_pic = glob.glob("CKplus_dataset//Image//{}//{}//*"
                                    .format(person, current))[-1]

            filenumber += 1
            dest_neut = ("combined_dataset//neutral//{}_{}_{}.png"
                         .format(dataset, filenumber, "frontal"))
            filenumber += 1
            dest_emot = ("combined_dataset//{}//{}_{}_{}.png"
                         .format(EMOTIONS_8[emotion], dataset,
                                 filenumber, "frontal"))

            try:
                # All images are kept to a standardised format.
                standardise_image(neutral_pic)
                standardise_image(emotion_pic)

                # Finally, copy the files to the combined dataset.
                copyfile(neutral_pic, dest_neut)
                copyfile(emotion_pic, dest_emot)

                print("Successful copy number {} for {} dataset."
                      .format((filenumber-1), dataset))
                print("Successful copy number {} for {} dataset."
                      .format(filenumber, dataset))
            except OSError as e:
                print('***> Some IO error occurred!!')
                continue


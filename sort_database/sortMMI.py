#!/usr/bin/env python3

"""Python script to sort the MMI Facial Expression Database.

The MMI dataset images have been analysed using Amazon Mechanical Turk, and
sorted into folders within 'MMI_Dataset' using another script, included in
the 'MMI_Dataset' folder.

Via https://mmifacedb.eu/:
    'The database consists of over 2900 videos and high-resolution still
    images of 75 subjects. It is fully annotated for the presence of AUs
    in videos (event coding), and partially coded on frame-level, indicating
    for each frame whether an AU is in either the neutral, onset, apex or
    offset phase. A small part was annotated for audio-visual laughters.
    The database is freely available to the scientific community.'

I found the Action Units were fairly inconsistent. Therefore I used Amazon's
worker service 'Mechanical Turk' to classify the images via a questionaire.

TODO : This is still stupid.

I requested an account and was granted permission for use.
"""

# Import packages.
import csv
from shutil import copyfile
from pathlib import Path

# My imports.
from utils import standardise_image


dataset = "MMI"


print("***> Processing MMI Facial Expression Database...")

avg = 0
cnt = 0
temp = 0
imgName = ""
with open('MMI_dataset//batch.csv') as csvfile:  # Coded using Mechanical Turk
    reader = csv.DictReader(csvfile)
    for row in reader:
        temp = int(row['AvgRate'])
        avg += temp
        imgName = row['Input.image_url']
        fpath = "MMI_dataset//images//{0!s}".format(imgName)

        my_file = Path(fpath)
        if my_file.is_file():
            cnt += 1

            # Use the emotion declared in row to build destination string.
            if (row['Anger'] == "1"):
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("anger", dataset, cnt, "frontal")
            elif row['Contempt'] == "1":
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("contempt", dataset, cnt, "frontal")
            elif row['Disgust'] == "1":
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("disgust", dataset, cnt, "frontal")
            elif row['Fear'] == "1":
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("fear", dataset, cnt, "frontal")
            elif row['Happy'] == "1":
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("happy", dataset, cnt, "frontal")
            elif row['Sadness'] == "1":
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("sadness", dataset, cnt, "frontal")
            elif row['Surprise'] == "1":
                dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("surprise", dataset, cnt, "frontal")

            # All images are kept to a standardised format.
            standardise_image(fpath)

            # Finally, copy the files to the combined dataset.
            copyfile(fpath, dest_emot)

            print("Successful copy number {} for {} dataset."
                  .format(cnt, dataset))
        # else:
        #     print("{}: File was deleted.".format(dataset))

with open('MMI_turk_accuracy', "w") as text_file:
    print("***> Average Ranking of Amazon Mechanical Turk accuracy is {}%."
          .format(int(avg/cnt)))

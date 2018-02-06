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

I requested an account and was granted permission for use.
"""
from PIL import Image
from shutil import copyfile
from pathlib import Path

import csv
import PIL


dataset = "MMI"


def standardise_image(pic):
    """Save image in resized, standard format."""
    try:
        img = Image.open(open(pic, 'rb')).convert('LA')  # Grayscale
    except IOError:
        print("File open error occurred.")

    basewidth = 480  # Resize
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    img.save(pic, "PNG")  # Save as .png


print("***> Processing MMI Facial Expression Database...")

avg = 0
cnt = 0
temp = 0
imgName = ""
with open('MMI_dataset//batch.csv') as csvfile:  # Coded using Mechanical Turk
    reader = csv.DictReader(csvfile)
    for row in reader:
        cnt += 1
        temp = int(row['AvgRate'])
        avg += temp
        imgName = row['Input.image_url']
        fpath = "MMI_dataset//images//{}".format(imgName)

        my_file = Path(fpath)
        if my_file.is_file():
            # Use the emotion declared in row to build destination string.
            if row['Anger'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("anger", cnt)
            elif row['Contempt'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("contempt", cnt)
            elif row['Disgust'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("disgust", cnt)
            elif row['Fear'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("fear", cnt)
            elif row['Happy'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("happy", cnt)
            elif row['Sadness'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("sadness", cnt)
            elif row['Surprise'] == "1":
                dest_emot = "combined_dataset//{}//MMI{}.png".format("surprise", cnt)

            # All images are kept to a standardised format.
            standardise_image(fpath)

            # Finally, copy the files to the combined dataset.
            copyfile(fpath, dest_emot)

            print("{}: Using 'batch.csv' - Copied {} to {}."
                  .format(dataset, imgName.replace("//", "/"),
                          dest_emot.replace("//", "/")))
        else:
            print("{}: File was deleted.".format(dataset))

print("\n***> Average Ranking of Amazon Mechanical Turk accurancy is {} percent."
      .format(int(avg/cnt)))

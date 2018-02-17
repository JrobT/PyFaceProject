#!/usr/bin/env python3

"""Python script to sort the JAFFE Database.

Via http://www.kasrl.org/jaffe_info.html:
    'Contains over 60 Japanese female subjects.'

I agreed to site the database in my report.
"""

# Import packages.
import csv
import os
from shutil import copyfile

# My imports.
from utils import standardise_image


dataset = "JAFFE"


print("***> Processing JAFFE Database...")

# Convert .txt into .csv. May need to change the first line of .txt file.
with open('JAFFE_dataset//JAFFE.txt') as fin, open("""JAFFE_dataset//
                                                   JAFFE.csv""", 'w') as fout:
    o = csv.writer(fout)
    for line in fin:
        o.writerow(line.split())

filenumber = 0
# Move JAFFE images into respective sets.
with open('JAFFE_dataset//JAFFE.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filenumber += 1
        lst = [row['HAP'], row['SAD'], row['SUR'],
               row['ANG'], row['DIS'], row['FEA']]
        maxx = max(lst)
        index = lst.index(maxx)  # Get the emotion most picked

        if (float(maxx) >= 3):  # less than 3 means the emotion is 'undecided'

            imgName = row['PIC']
            fpath = 'JAFFE_dataset//images//{0!s}'.format(imgName)

            if (os.path.exists(fpath)):
                if (index == 0):
                    dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("happy", dataset, filenumber, "frontal")
                elif (index == 1):
                    dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("sadness", dataset, filenumber, "frontal")
                elif index == 2:
                    dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("surprise", dataset, filenumber, "frontal")
                elif index == 3:
                    dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("anger", dataset, filenumber, "frontal")
                elif index == 4:
                    dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("disgust", dataset, filenumber, "frontal")
                elif index == 5:
                    dest_emot = "combined_dataset//{}//{}_{}_{}.png".format("fear", dataset, filenumber, "frontal")

                try:
                    # All images are kept to a standardised format.
                    standardise_image(fpath)

                    # Finally, copy the files to the combined dataset.
                    copyfile(fpath, dest_emot)

                    print("Successful copy number {} for {} dataset."
                          .format(filenumber, dataset))
                except OSError as e:
                    print('***> Some IO error occurred!!')
                    continue
        #     else:
        #         print("{}: Using 'JAFFE.csv' - File doesn't exist."
        #               .format(dataset))
        # else:
        #     print("{}: Using 'JAFFE.csv' - The emotion is unclear."
        #           .format(dataset))

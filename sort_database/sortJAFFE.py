#!/usr/bin/env python3
"""Python script to sort the JAFFE Database.

Via http://www.kasrl.org/jaffe_info.html:
    'Contains over 60 Japanese female subjects.'

I agreed to site the database in my report.
"""
import csv
import PIL
import os
from PIL import Image
from shutil import copyfile


dataset = "JAFFE"


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = Image.open(open(pic, 'rb')).convert('LA')  # Grayscale

    basewidth = 480  # Resize
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    img.save(pic, "PNG")  # Save as .png


print("***> Processing JAFFE Database...")

filenumber = 0

# Convert .txt into .csv. May need to change the first line of .txt file.
with open('JAFFE_Dataset//JAFFE.txt') as fin, open("""JAFFE_dataset//
                                                   JAFFE.csv""", 'w') as fout:
    o = csv.writer(fout)
    for line in fin:
        o.writerow(line.split())

# Move JAFFE images into respective sets.
with open('JAFFE_dataset//JAFFE.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filenumber += 1
        lst = [row['HAP'], row['SAD'], row['SUR'],
               row['ANG'], row['DIS'], row['FEA']]
        maxx = max(lst)
        index = lst.index(maxx)  # Get the emotion most picked

        if (float(maxx) >= 3):  # less than 3 means the emotion is 'undecided' IMHO

            imgName = row['PIC']
            fpath = 'JAFFE_dataset//images//{0!s}'.format(imgName)

            if (os.path.exists(fpath)):
                if index == 0:
                    dest_emot = "combined_dataset//{}//JAFFE{}.png".format("happy",
                                                                           filenumber)
                elif index == 1:
                    dest_emot = "combined_dataset//{}//JAFFE{}.png".format("sadness",
                                                                           filenumber)
                elif index == 2:
                    dest_emot = "combined_dataset//{}//JAFFE{}.png".format("surprise",
                                                                           filenumber)
                elif index == 3:
                    dest_emot = "combined_dataset//{}//JAFFE{}.png".format("anger",
                                                                           filenumber)
                elif index == 4:
                    dest_emot = "combined_dataset//{}//JAFFE{}.png".format("disgust",
                                                                           filenumber)
                elif index == 5:
                    dest_emot = "combined_dataset//{}//JAFFE{}.png".format("fear",
                                                                           filenumber)

                try:
                    # All images are kept to a standardised format.
                    standardise_image(fpath)

                    # Finally, copy the files to the combined dataset.
                    copyfile(fpath, dest_emot)

                    print("{}: Using 'JAFFE.csv' - Copied {} into {}."
                          .format(dataset, imgName.replace("//", "/"),
                                  dest_emot.replace("//", "/")))
                except OSError as e:
                    print('***> Some IO error occurred!!')
                    continue
            else:
                print("{}: Using 'JAFFE.csv' - File doesn't exist."
                      .format(dataset))
        else:
            print("{}: Using 'JAFFE.csv' - The emotion is unclear."
                  .format(dataset))

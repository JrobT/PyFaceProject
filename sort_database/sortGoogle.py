#!/usr/bin/env python3
"""Python script to sort the Google Images Dataset.

I filled out my database by using the Google images search engine. By typing
each emotion into the search, I used the ZIG Lite Chrome extension
(chrome-extension://bedbigoemkinkepgmcmgnapjcahnedmn/out.html?get) to scrape
the page for suitable images.

This may not be strictly useful. These images come without proper permission.
"""
import glob
import PIL
from PIL import Image
from shutil import copyfile


dataset = "Google Images"


def standardise_image(pic):
    """Save image in resized, standard format."""
    img = Image.open(open(pic, 'rb')).convert('LA')  # Grayscale

    basewidth = 480  # Resize
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    img.save(pic, "PNG")  # Save as .png


print("***> Processing my `Google Images' Dataset...".format(dataset))

emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]  # The emotion list

num = 0
for emotion in emotions:
    for f in glob.glob("google_dataset//%s//*" % emotion):
        num = num + 1
        dest = "combined_dataset//%s//Google%s.png" % (emotion, num)

        try:
            # All images are kept to a standardised format.
            standardise_image(f)

            # Finally, copy the files to the combined dataset.
            copyfile(f, dest)

            print("{}: Copied {} into {}.".format(dataset,
                                                  f.replace("//", "/"),
                                                  dest.replace("//", "/")))
        except OSError as e:
            print('***> Some IO error occurred!!')
            continue
